import numpy as np
from datetime import datetime
from typing import Tuple
from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

class OrbitCalculator:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.earth_radius = 6371  # 地球半径(km)
        
    def calculate_satellite_position(self, sat_config: 'SatelliteConfig', time: datetime) -> Tuple[float, float, float]:
        """计算卫星在给定时间的位置
        
        参数:
            sat_config: 卫星配置
            time: 计算时间点
            
        返回:
            (x, y, z): 卫星在地心惯性坐标系中的位置(km)
        """
        if self.debug_mode:
            return (0.0, 0.0, 0.0)
            
        # 创建轨道对象
        orb = Orbit.from_classical(
            Earth,
            sat_config.semi_major_axis * u.km,
            sat_config.eccentricity * u.one,
            sat_config.inclination * u.deg,
            sat_config.raan * u.deg,
            sat_config.arg_perigee * u.deg,
            0 * u.rad,  # 真近点角
            epoch=Time(time)
        )
        
        # 获取位置向量
        r = orb.r
        return (r[0].to(u.km).value, r[1].to(u.km).value, r[2].to(u.km).value)
        
    def calculate_satellite_distance(self, sat1_config: 'SatelliteConfig', sat2_config: 'SatelliteConfig') -> float:
        """计算两颗卫星之间的距离(km)"""
        if self.debug_mode:
            return 0.0
            
        # 获取当前时间
        current_time = datetime.now()
        
        # 计算两颗卫星的位置
        pos1 = self.calculate_satellite_position(sat1_config, current_time)
        pos2 = self.calculate_satellite_position(sat2_config, current_time)
        
        # 计算欧氏距离
        return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))
        
    def check_communication_window(self, sat1_config: 'SatelliteConfig', sat2_config: 'SatelliteConfig') -> bool:
        """检查两颗卫星是否在通信窗口内"""
        if self.debug_mode:
            return sat1_config.orbit_id == sat2_config.orbit_id
            
        try:
            # 计算卫星间距离
            distance = self.calculate_satellite_distance(sat1_config, sat2_config)
            
            # 检查视线通信条件
            pos1 = self.calculate_satellite_position(sat1_config, datetime.now())
            pos2 = self.calculate_satellite_position(sat2_config, datetime.now())
            
            # 检查地球遮挡
            if self._check_earth_obstruction(pos1, pos2):
                return False
            
            # 检查最大通信距离
            max_comm_distance = sat1_config.max_communication_distance
            return bool(distance <= max_comm_distance)  # 显式转换为布尔值
            
        except Exception as e:
            print(f"通信窗口检查失败: {str(e)}")
            return False
        
    def _check_earth_obstruction(self, pos1: Tuple[float, float, float], 
                                pos2: Tuple[float, float, float]) -> bool:
        """检查地球是否遮挡了两颗卫星间的通信
        
        使用几何方法检查连线是否穿过地球
        """
        # 将位置向量转换为numpy数组
        p1 = np.array(pos1)
        p2 = np.array(pos2)
        
        # 计算连线向量
        d = p2 - p1
        
        # 解二次方程判断是否与地球相交
        a = np.dot(d, d)
        b = 2 * np.dot(p1, d)
        c = np.dot(p1, p1) - self.earth_radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant <= 0:
            return False
            
        # 计算交点参数
        t1 = (-b + np.sqrt(discriminant)) / (2*a)
        t2 = (-b - np.sqrt(discriminant)) / (2*a)
        
        # 如果有交点在[0,1]区间内，说明有遮挡
        return (0 <= t1 <= 1) or (0 <= t2 <= 1) 