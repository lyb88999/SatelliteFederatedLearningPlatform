import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Union, TYPE_CHECKING

# 避免循环导入
if TYPE_CHECKING:
    from .ground_station import GroundStation
    from .config import SatelliteConfig
else:
    SatelliteConfig = 'SatelliteConfig'
    GroundStation = 'GroundStation'

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.util import time_range
from astropy.coordinates import CartesianRepresentation

class OrbitCalculator:
    """轨道计算器"""
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.earth_radius = 6371.0  # 地球半径(km)
        self.earth_mu = 398600.4418  # 地球引力常数(km³/s²)
        self.max_communication_distance = 5000.0  # 最大通信距离(km)
        
    def check_satellite_visibility(self, 
                                 satellite: SatelliteConfig,
                                 ground_station: GroundStation,
                                 current_time: datetime) -> bool:
        """检查卫星是否可见于地面站
        
        Args:
            satellite: 卫星配置
            ground_station: 地面站
            current_time: 当前时间
        
        Returns:
            bool: 是否可见
        """
        if self.debug_mode:
            # 在调试模式下，每个地面站可以看到2-3个轨道
            orbit_id = satellite.orbit_id
            station_id = ground_station.config.station_id
            
            visibility_map = {
                "Beijing": [0, 1],
                "NewYork": [1, 2],
                "London": [2, 3],
                "Sydney": [3, 4],
                "Moscow": [4, 5],
                "SaoPaulo": [5, 0]
            }
            
            return orbit_id in visibility_map.get(station_id, [])
            
        # 计算卫星位置
        sat_pos = self.calculate_satellite_position(
            satellite, current_time
        )
        
        # 计算地面站位置
        station_pos = self.calculate_ground_station_position(
            ground_station.config
        )
        
        # 计算距离
        distance = np.linalg.norm(sat_pos - station_pos)
        
        # 计算仰角
        elevation = self.calculate_elevation(
            sat_pos, station_pos
        )
        
        # 检查可见性条件
        return (distance <= ground_station.config.max_range and 
                elevation >= ground_station.config.min_elevation)
        
    def calculate_satellite_position(self, sat_config: SatelliteConfig, time: datetime) -> Tuple[float, float, float]:
        """计算卫星在给定时间的位置"""
        if self.debug_mode:
            # 在调试模式下生成一个更合理的分布
            phase = np.radians(sat_config.arg_perigee)
            inc = np.radians(sat_config.inclination)
            raan = np.radians(sat_config.raan)
            
            r = sat_config.semi_major_axis
            x_orbit = r * np.cos(phase)
            y_orbit = r * np.sin(phase)
            
            x = x_orbit * np.cos(raan) - y_orbit * np.sin(raan) * np.cos(inc)
            y = x_orbit * np.sin(raan) + y_orbit * np.cos(raan) * np.cos(inc)
            z = y_orbit * np.sin(inc)
            
            return (float(x), float(y), float(z))
            
        try:
            # 使用 poliastro 计算位置
            orbit = Orbit.from_classical(
                Earth,
                sat_config.semi_major_axis * u.km,
                sat_config.eccentricity * u.one,
                sat_config.inclination * u.deg,
                sat_config.raan * u.deg,
                sat_config.arg_perigee * u.deg,
                0 * u.deg,  # 真近点角，使用0作为初始值
                epoch=Time(sat_config.epoch)
            )
            
            # 计算给定时间的位置
            dt = (time - sat_config.epoch).total_seconds() * u.s
            pos = orbit.propagate(dt).r
            
            return (float(pos[0].value), float(pos[1].value), float(pos[2].value))
            
        except Exception as e:
            print(f"位置计算错误: {str(e)}")
            return (0.0, 0.0, 0.0)
            
    def calculate_satellite_distance(self, sat1: SatelliteConfig, sat2: SatelliteConfig, 
                                  time: datetime) -> float:
        """计算两颗卫星之间的距离"""
        pos1 = self.calculate_satellite_position(sat1, time)
        pos2 = self.calculate_satellite_position(sat2, time)
        return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))
        
    def calculate_elevation(self, station_pos: np.ndarray, sat_pos: np.ndarray) -> float:
        """计算仰角（度）"""
        # 计算地面站到卫星的向量
        r_vector = sat_pos - station_pos
        
        # 计算地面站的地心方向单位向量
        station_unit = station_pos / np.linalg.norm(station_pos)
        
        # 计算夹角
        cos_angle = np.dot(r_vector, station_unit) / np.linalg.norm(r_vector)
        angle = np.arccos(cos_angle)
        
        # 转换为仰角
        elevation = 90 - np.degrees(angle)
        return elevation

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