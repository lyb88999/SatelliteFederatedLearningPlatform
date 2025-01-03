from dataclasses import dataclass, field
from typing import List, Tuple, Union
from datetime import datetime

@dataclass
class GroundStationConfig:
    def __init__(
        self,
        station_id: str,
        latitude: float,
        longitude: float,
        max_range: float,
        min_elevation: float,
        max_satellites: int = 4  # 最大同时可见卫星数
    ):
        self.station_id = station_id
        self.latitude = latitude    # 纬度（度）
        self.longitude = longitude  # 经度（度）
        self.max_range = max_range  # 最大通信距离（km）
        self.min_elevation = min_elevation  # 最小仰角（度）
        self.max_satellites = max_satellites  # 最大同时可见卫星数

@dataclass
class SatelliteConfig:
    """卫星配置"""
    def __init__(self,
                 orbit_id: int,
                 sat_id: int,
                 semi_major_axis: float,
                 eccentricity: float,
                 inclination: float,
                 raan: float,
                 arg_perigee: float,
                 epoch: datetime):
        self.orbit_id = orbit_id
        self.sat_id = sat_id
        self.semi_major_axis = semi_major_axis
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.raan = raan
        self.arg_perigee = arg_perigee
        self.epoch = epoch
        self.max_communication_distance: float = 1000.0
        self.ground_stations: List[GroundStationConfig] = field(default_factory=list)
        self.is_coordinator: bool = False 