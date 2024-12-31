import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
from .config import SatelliteConfig, GroundStationConfig

class OrbitCalculator:
    def __init__(self, satellite_config: SatelliteConfig, debug_mode: bool = True):
        self.config = satellite_config
        self.earth_radius = 6371  # 地球半径（公里）
        self.orbit_radius = self.earth_radius + self.config.orbit_altitude
        self.debug_mode = debug_mode
        
    def _calculate_satellite_position(self, time: datetime) -> Tuple[float, float, float]:
        """计算卫星在给定时间的位置"""
        # 计算时间参数
        t = time.timestamp()
        t0 = datetime(time.year, time.month, time.day).timestamp()
        dt = t - t0
        
        # 计算轨道角度
        n = 2 * np.pi / (self.config.orbital_period * 60)  # 平均运动（弧度/秒）
        orbit_angle = n * dt  # 轨道角度
        
        # 1. 计算卫星在轨道平面内的位置（相对于地心）
        x = self.orbit_radius * np.cos(orbit_angle)
        y = self.orbit_radius * np.sin(orbit_angle)
        z = 0.0
        
        # 2. 应用轨道倾角（绕X轴旋转）
        inclination = np.radians(self.config.orbit_inclination)
        y_inclined = y * np.cos(inclination)
        z_inclined = y * np.sin(inclination)
        
        # 3. 应用升交点赤经（绕Z轴旋转）
        raan = np.radians(self.config.ascending_node)
        x_final = x * np.cos(raan) - y_inclined * np.sin(raan)
        y_final = x * np.sin(raan) + y_inclined * np.cos(raan)
        z_final = z_inclined
        
        return (x_final, y_final, z_final)
    
    def _calculate_ground_station_position(self, station: GroundStationConfig, time: datetime) -> Tuple[float, float, float]:
        """计算地面站的位置"""
        lat = np.radians(station.latitude)
        lon = np.radians(station.longitude)
        
        # 计算地方恒星时（考虑地球自转）
        t = time.timestamp()
        t0 = datetime(time.year, time.month, time.day).timestamp()
        dt = t - t0
        sid_time = 2 * np.pi * (dt % 86400) / 86400
        lon_adjusted = lon + sid_time
        
        # 计算地面站的笛卡尔坐标
        x = self.earth_radius * np.cos(lat) * np.cos(lon_adjusted)
        y = self.earth_radius * np.cos(lat) * np.sin(lon_adjusted)
        z = self.earth_radius * np.sin(lat)
        
        return (x, y, z)
    
    def calculate_visibility(self, ground_station: GroundStationConfig, time: datetime) -> bool:
        """计算卫星是否可见"""
        if self.debug_mode:
            # 在调试模式下，每30秒都有15秒的通信窗口
            seconds = time.timestamp()
            return (int(seconds) % 30) < 15
            
        # 正常模式下使用真实的轨道计算
        return self._calculate_real_visibility(ground_station, time)
        
    def _calculate_real_visibility(self, ground_station: GroundStationConfig, time: datetime) -> bool:
        """计算真实的可见性"""
        # 原来的可见性计算代码
        elevation, slant_range = self._get_pass_metrics(ground_station, time)
        return (elevation >= ground_station.min_elevation and 
                slant_range <= ground_station.coverage_radius)
    
    def get_next_window(self, ground_station: GroundStationConfig, 
                       current_time: datetime) -> Tuple[datetime, timedelta]:
        """计算下一个通信窗口的开始时间和持续时间"""
        if self.debug_mode:
            # 在调试模式下，窗口周期为30秒
            seconds = current_time.timestamp()
            current_mod = int(seconds) % 30
            if current_mod < 15:
                # 已经在窗口内
                return current_time, timedelta(seconds=15-current_mod)
            else:
                # 等待下一个窗口
                wait_seconds = 30 - current_mod
                next_window = current_time + timedelta(seconds=wait_seconds)
                return next_window, timedelta(seconds=15)
        
        # 正常模式下使用真实的轨道计算
        return self._calculate_real_next_window(ground_station, current_time)
        
    def _calculate_real_next_window(self, ground_station: GroundStationConfig, 
                                  current_time: datetime) -> Tuple[datetime, timedelta]:
        """计算真实的下一个通信窗口"""
        # 原来的窗口预测代码
        window_start = None
        window_duration = timedelta(minutes=10)
        
        # 向前搜索24小时内的下一个窗口
        for minutes in range(24 * 60):
            test_time = current_time + timedelta(minutes=minutes)
            if self._calculate_real_visibility(ground_station, test_time):
                window_start = test_time
                break
                
        return window_start, window_duration 