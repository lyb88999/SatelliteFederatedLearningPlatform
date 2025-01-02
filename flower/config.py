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
        orbit_altitude: float = 550.0,
        orbit_inclination: float = 97.6,
        orbital_period: float = 95.0,
        ground_stations: List[GroundStationConfig] = None,
        ascending_node: float = 0.0,
        mean_anomaly: float = 0.0,
        orbit_id: int = 0,
        sat_id: int = 0,
        is_coordinator: bool = False
    ):
        self.orbit_altitude = orbit_altitude
        self.orbit_inclination = orbit_inclination
        self.orbital_period = orbital_period
        self.ground_stations = ground_stations or []
        self.ascending_node = ascending_node
        self.mean_anomaly = mean_anomaly
        self.orbit_id = orbit_id
        self.sat_id = sat_id
        self.is_coordinator = is_coordinator 