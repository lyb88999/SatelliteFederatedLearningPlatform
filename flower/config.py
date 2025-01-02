from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import datetime

@dataclass
class GroundStationConfig:
    station_id: str
    latitude: float
    longitude: float
    coverage_radius: float = 3000  # 增加覆盖半径
    min_elevation: float = 5.0  # 降低最小仰角要求

@dataclass
class SatelliteConfig:
    orbit_id: int
    sat_id: int
    is_coordinator: bool = False
    
    # 轨道参数
    semi_major_axis: float = 7000.0  # 半长轴(km)
    eccentricity: float = 0.0        # 偏心率
    inclination: float = 98.0        # 倾角(度)
    raan: float = 0.0                # 升交点赤经(度)
    arg_perigee: float = 0.0         # 近地点幅角(度)
    epoch: datetime = field(default_factory=datetime.now)  # 历元
    
    # 通信参数
    max_communication_distance: float = 1000.0  # 最大通信距离(km) 