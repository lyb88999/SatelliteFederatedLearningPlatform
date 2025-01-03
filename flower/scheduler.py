from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict
from .config import SatelliteConfig
from .orbit_utils import OrbitCalculator

@dataclass
class CommunicationWindow:
    """通信窗口"""
    start_time: datetime
    end_time: datetime
    satellite: SatelliteConfig
    quality: float  # 链路质量（0-1）

@dataclass
class Task:
    """任务定义"""
    task_id: str
    satellite_id: Optional[str]  # 可以为 None，等待分配
    start_time: datetime
    duration: float  # 秒
    priority: int = 1
    status: str = "pending"  # pending, running, completed, failed
    station_id: Optional[str] = None  # 添加这行
    deadline: Optional[datetime] = None  # 添加这行

class Scheduler:
    """任务调度器"""
    def __init__(self, orbit_calculator: OrbitCalculator):
        self.orbit_calculator = orbit_calculator
        self.tasks: List[Task] = []
        self.windows: Dict[str, List[CommunicationWindow]] = {}
        
    def add_task(self, task: Task):
        """添加任务到调度器"""
        self.tasks.append(task)
        
    def get_communication_windows(self, satellite: SatelliteConfig, 
                                start_time: datetime, duration: float) -> List[CommunicationWindow]:
        """获取通信窗口"""
        # TODO: 实现通信窗口计算
        return []
        
    def schedule_tasks(self) -> List[Task]:
        """调度任务"""
        # 按优先级排序
        self.tasks.sort(key=lambda x: (-x.priority, x.start_time))
        return self.tasks
        
    def update_task_status(self, task_id: str, status: str):
        """更新任务状态"""
        for task in self.tasks:
            if task.task_id == task_id:
                task.status = status
                break 