from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .config import SatelliteConfig, GroundStationConfig
from .orbit_utils import OrbitCalculator
import numpy as np

@dataclass
class CommunicationWindow:
    station_id: str
    start_time: datetime
    end_time: datetime
    max_elevation: float
    min_distance: float

@dataclass
class Task:
    task_id: str
    duration: int  # 任务持续时间（秒）
    priority: int  # 优先级（1-10）
    station_id: str
    deadline: datetime

class Scheduler:
    def __init__(self, satellite_config: SatelliteConfig):
        self.config = satellite_config
        self.orbit_calculator = OrbitCalculator(satellite_config)
        self.tasks: List[Task] = []
        
    def predict_windows(self, start_time: datetime, duration_hours: int = 24) -> List[CommunicationWindow]:
        """预测未来时间段内的所有通信窗口"""
        windows = []
        
        for station in self.config.ground_stations:
            current_window = None
            max_elevation = 0
            min_distance = float('inf')
            
            # 按分钟检查可见性
            for minutes in range(duration_hours * 60):
                check_time = start_time + timedelta(minutes=minutes)
                is_visible = self.orbit_calculator.calculate_visibility(station, check_time)
                
                if is_visible:
                    # 获取仰角和距离
                    elevation, distance = self._get_pass_metrics(station, check_time)
                    
                    if current_window is None:
                        # 新窗口开始
                        current_window = CommunicationWindow(
                            station_id=station.station_id,
                            start_time=check_time,
                            end_time=check_time + timedelta(minutes=1),  # 初始设置为1分钟后
                            max_elevation=elevation,
                            min_distance=distance
                        )
                    else:
                        # 更新现有窗口
                        current_window.end_time = check_time + timedelta(minutes=1)  # 延长到下一分钟
                        current_window.max_elevation = max(current_window.max_elevation, elevation)
                        current_window.min_distance = min(current_window.min_distance, distance)
                
                elif current_window is not None:
                    # 窗口结束，检查持续时间
                    duration = (current_window.end_time - current_window.start_time).total_seconds() / 60
                    if duration >= 5:  # 只保存持续时间大于等于5分钟的窗口
                        windows.append(current_window)
                    current_window = None
                    max_elevation = 0
                    min_distance = float('inf')
        
        # 检查最后一个窗口
        if current_window is not None:
            duration = (current_window.end_time - current_window.start_time).total_seconds() / 60
            if duration >= 5:
                windows.append(current_window)
        
        return sorted(windows, key=lambda w: w.start_time)
    
    def add_task(self, task: Task) -> bool:
        """添加新任务到调度队列"""
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: t.priority, reverse=True)  # 按优先级排序
        return True
    
    def schedule_tasks(self, start_time: datetime, duration_hours: int = 24) -> Dict[str, List[Tuple[Task, datetime]]]:
        """为任务分配通信窗口"""
        windows = self.predict_windows(start_time, duration_hours)
        schedule = {}  # station_id -> [(task, scheduled_time), ...]
        
        # 为每个地面站创建空的调度列表
        for station in self.config.ground_stations:
            schedule[station.station_id] = []
        
        # 按优先级处理任务
        for task in self.tasks:
            assigned = False
            
            # 查找合适的窗口
            for window in windows:
                if (window.station_id == task.station_id and 
                    window.start_time < task.deadline and
                    (window.end_time - window.start_time).seconds >= task.duration):
                    
                    # 分配任务到这个窗口
                    schedule[task.station_id].append((task, window.start_time))
                    assigned = True
                    break
            
            if not assigned:
                print(f"Warning: Could not schedule task {task.task_id}")
        
        return schedule
    
    def _get_pass_metrics(self, station: GroundStationConfig, time: datetime) -> Tuple[float, float]:
        """获取特定时间点的通过指标（仰角和距离）"""
        sat_pos = self.orbit_calculator._calculate_satellite_position(time)
        gs_pos = self.orbit_calculator._calculate_ground_station_position(station, time)
        
        # 计算距离
        dx = sat_pos[0] - gs_pos[0]
        dy = sat_pos[1] - gs_pos[1]
        dz = sat_pos[2] - gs_pos[2]
        distance = (dx*dx + dy*dy + dz*dz) ** 0.5
        
        # 计算仰角
        gs_pos_array = np.array(gs_pos)
        zenith = gs_pos_array / self.orbit_calculator.earth_radius
        
        # 将向量转换为numpy数组
        los = np.array([dx, dy, dz]) / distance
        elevation = np.degrees(np.arcsin(np.dot(zenith, los)))
        
        return elevation, distance 