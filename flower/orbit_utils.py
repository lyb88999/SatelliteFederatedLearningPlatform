import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Union, TYPE_CHECKING

# 避免循环导入
if TYPE_CHECKING:
    from .ground_station import GroundStation
    from .config import SatelliteConfig, GroundStationConfig
else:
    SatelliteConfig = 'SatelliteConfig'
    GroundStation = 'GroundStation'
    GroundStationConfig = 'GroundStationConfig'

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.coordinates import CartesianRepresentation

# 移除 from poliastro.util import time_range
# 替代方案：自己实现 time_range 函数
def time_range(start_time: datetime, end_time: datetime, step: timedelta) -> List[datetime]:
    """生成时间序列
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        step: 时间步长
        
    Returns:
        List[datetime]: 时间序列
    """
    times = []
    current = start_time
    while current <= end_time:
        times.append(current)
        current += step
    return times

class OrbitCalculator:
    """轨道计算器"""
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.earth_radius = 6371.0  # 地球半径(km)
        self.earth_mu = 398600.4418  # 地球引力常数(km³/s²)
        self.max_communication_distance = 5000.0  # 最大通信距离(km)
        
    def check_satellite_visibility(self, 
                                 satellite: 'SatelliteConfig',
                                 ground_station: Union['GroundStation', 'GroundStationConfig'],
                                 current_time: datetime) -> bool:
        """检查卫星是否可见于地面站"""
        config = ground_station.config if hasattr(ground_station, 'config') else ground_station
        
        # 计算卫星位置和地面站位置
        sat_pos = np.array(self.calculate_satellite_position(satellite, current_time))
        station_pos = self.calculate_ground_station_position(config)
        
        # 计算距离和仰角
        distance = np.linalg.norm(sat_pos - station_pos)
        elevation = self._calculate_elevation(sat_pos, station_pos)
        
        if self.debug_mode:
            print(f"Distance: {distance:.1f} km")
            print(f"Elevation: {elevation:.1f}°")
            print(f"Max range: {config.max_range} km")
            print(f"Min elevation: {config.min_elevation}°")
        
        # 检查可见性条件
        is_visible = (
            distance <= config.max_range and  # 在最大通信距离内
            elevation >= config.min_elevation  # 满足最小仰角要求
        )
        
        return is_visible
        
    def calculate_satellite_position(self, sat_config: SatelliteConfig, time: datetime) -> Tuple[float, float, float]:
        """计算卫星在给定时间的位置"""
        try:
            # 计算从历元开始经过的时间（秒）
            dt = (time - sat_config.epoch).total_seconds()
            
            # 计算轨道周期（秒）
            orbital_period = 2 * np.pi * np.sqrt(sat_config.semi_major_axis**3 / self.earth_mu)
            
            # 计算当前相位角（考虑初始相位和时间演化）
            phase = np.radians(sat_config.arg_perigee) + (2 * np.pi * dt / orbital_period)
            inc = np.radians(sat_config.inclination)
            raan = np.radians(sat_config.raan)
            
            # 计算轨道平面内的位置
            r = sat_config.semi_major_axis
            x_orbit = r * np.cos(phase)
            y_orbit = r * np.sin(phase)
            
            # 转换到地心惯性坐标系
            x = x_orbit * np.cos(raan) - y_orbit * np.sin(raan) * np.cos(inc)
            y = x_orbit * np.sin(raan) + y_orbit * np.cos(raan) * np.cos(inc)
            z = y_orbit * np.sin(inc)
            
            return (float(x), float(y), float(z))
            
        except Exception as e:
            print(f"位置计算错误: {str(e)}")
            return (0.0, 0.0, 0.0)
            
    def calculate_satellite_distance(self, sat1: SatelliteConfig, sat2: SatelliteConfig, 
                                  time: datetime) -> float:
        """计算两颗卫星之间的距离"""
        pos1 = self.calculate_satellite_position(sat1, time)
        pos2 = self.calculate_satellite_position(sat2, time)
        return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))
        
    def calculate_orbital_period(self, semi_major_axis: float) -> float:
        """计算轨道周期（秒）"""
        return 2 * np.pi * np.sqrt(semi_major_axis**3 / self.earth_mu)

    def check_communication_window(self, sat1: SatelliteConfig, sat2: SatelliteConfig) -> bool:
        """检查两颗卫星是否在通信窗口内"""
        return self.check_satellite_visibility(sat1, sat2, datetime.now())

    def _calculate_satellite_position(self, time: datetime) -> np.ndarray:
        """内部方法：计算卫星位置"""
        if self.current_satellite is None:
            # 如果没有设置当前卫星，返回原点
            return np.array([0.0, 0.0, 0.0])
        return np.array(self.calculate_satellite_position(self.current_satellite, time))

    def schedule_adaptive(self, satellites: List[SatelliteConfig], start_time: datetime,
                         resource_states: Dict, task_priorities: Dict, 
                         duration_hours: int) -> Dict:
        """自适应调度"""
        schedule = {}
        for sat in satellites:
            sat_id = f"orbit_{sat.orbit_id}_sat_{sat.sat_id}"
            schedule[sat_id] = []
        return schedule

    def select_best_coordinator(self, satellites: List[SatelliteConfig], 
                              start_time: datetime, duration_hours: int) -> Dict:
        """选择最佳协调者"""
        coordinator_scores = {}
        for sat in satellites:
            orbit_id = sat.orbit_id
            if orbit_id not in coordinator_scores:
                coordinator_scores[orbit_id] = (f"orbit_{orbit_id}_sat_{sat.sat_id}", 1.0)
        return coordinator_scores 

    def calculate_ground_track(self, sat_config: SatelliteConfig, 
                             start_time: datetime, duration_hours: float) -> List[Tuple[float, float]]:
        """计算地面轨迹
        
        Args:
            sat_config: 卫星配置
            start_time: 开始时间
            duration_hours: 持续时间（小时）
            
        Returns:
            List[Tuple[float, float]]: 经纬度列表 [(lon1, lat1), (lon2, lat2), ...]
        """
        if self.debug_mode:
            return [(0.0, 0.0), (10.0, 10.0)]  # 调试模式返回固定值
            
        track_points = []
        time_step = duration_hours * 3600 / 100  # 将时间分成100个点
        
        for i in range(100):
            current_time = start_time + timedelta(seconds=i * time_step)
            pos = self.calculate_satellite_position(sat_config, current_time)
            
            # 将笛卡尔坐标转换为经纬度
            x, y, z = pos
            r = np.sqrt(x*x + y*y + z*z)
            lat = np.arcsin(z/r) * 180/np.pi
            lon = np.arctan2(y, x) * 180/np.pi
            
            track_points.append((lon, lat))
            
        return track_points 

    def calculate_ground_station_position(self, station_config: 'GroundStationConfig') -> np.ndarray:
        """计算地面站的笛卡尔坐标
        
        Args:
            station_config: 地面站配置
            
        Returns:
            np.ndarray: [x, y, z] 笛卡尔坐标
        """
        # 将经纬度转换为弧度
        lat = np.radians(station_config.latitude)
        lon = np.radians(station_config.longitude)
        
        # 计算地面站的笛卡尔坐标
        x = self.earth_radius * np.cos(lat) * np.cos(lon)
        y = self.earth_radius * np.cos(lat) * np.sin(lon)
        z = self.earth_radius * np.sin(lat)
        
        return np.array([x, y, z]) 

    def _calculate_elevation(self, sat_pos: np.ndarray, station_pos: np.ndarray) -> float:
        """计算仰角（度）"""
        # 计算地面站到卫星的向量
        r_vector = sat_pos - station_pos
        
        # 计算地面站的地心方向单位向量
        station_unit = station_pos / np.linalg.norm(station_pos)
        
        # 计算夹角
        cos_angle = np.dot(r_vector, station_unit) / np.linalg.norm(r_vector)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 添加clip避免数值误差
        
        # 转换为仰角
        elevation = 90 - np.degrees(angle)
        return elevation

    def _check_earth_obstruction(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """检查地球是否遮挡了两点之间的视线
        
        Args:
            pos1: 第一个点的位置向量 [x, y, z]
            pos2: 第二个点的位置向量 [x, y, z]
            
        Returns:
            bool: True 如果地球遮挡了视线，False 否则
        """
        # 计算两点之间的向量
        r = pos2 - pos1
        dist = np.linalg.norm(r)
        
        # 计算pos1到地心的距离
        r1 = np.linalg.norm(pos1)
        
        # 如果pos1在地球表面，计算与地平面的夹角
        if abs(r1 - self.earth_radius) < 1.0:  # 允许1km的误差
            # 计算地平面法向量（指向天顶）
            zenith = pos1 / r1
            # 计算视线向量的单位向量
            sight = r / dist
            # 计算夹角的余弦值
            cos_angle = np.dot(zenith, sight)
            # 如果夹角大于90度，说明视线被地球遮挡
            if cos_angle < 0:
                return True
        
        # 计算视线到地心的最短距离
        # 使用向量代数计算点到直线的最短距离
        p = np.dot(pos1, r) / dist
        d = np.sqrt(np.dot(pos1, pos1) - p*p)
        
        # 如果最短距离小于地球半径，说明地球遮挡了视线
        return d < self.earth_radius 