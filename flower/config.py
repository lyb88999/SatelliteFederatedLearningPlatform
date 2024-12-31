from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GroundStationConfig:
    station_id: str
    latitude: float
    longitude: float
    coverage_radius: float = 3000  # 增加覆盖半径
    min_elevation: float = 5.0  # 降低最小仰角要求

@dataclass
class SatelliteConfig:
    def __init__(
        self,
        orbit_altitude: float,
        orbit_inclination: float,
        orbital_period: float,
        ground_stations: List[GroundStationConfig],
        ascending_node: float,
        mean_anomaly: float,
        orbit_id: int,           # 新增：轨道ID
        is_coordinator: bool     # 新增：是否为协调者节点
    ):
        self.orbit_altitude = orbit_altitude
        self.orbit_inclination = orbit_inclination
        self.orbital_period = orbital_period
        self.ground_stations = ground_stations
        self.ascending_node = ascending_node
        self.mean_anomaly = mean_anomaly
        self.orbit_id = orbit_id
        self.is_coordinator = is_coordinator 