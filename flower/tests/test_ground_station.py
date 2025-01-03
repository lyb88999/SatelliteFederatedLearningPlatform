import pytest
from datetime import datetime
from flower.config import SatelliteConfig, GroundStationConfig
from flower.ground_station import GroundStation
from flower.orbit_utils import OrbitCalculator

def test_ground_station_visibility():
    """测试地面站可见性检查"""
    orbit_calculator = OrbitCalculator(debug_mode=False)
    earth_radius = orbit_calculator.earth_radius
    
    station = GroundStation(
        GroundStationConfig(
            station_id="Beijing",
            latitude=39.9042,
            longitude=116.4074,
            max_range=1000.0,    # 1000km通信距离
            min_elevation=10.0,   # 10度最小仰角
            max_satellites=4
        ),
        orbit_calculator
    )
    
    satellites = []
    for i in range(4):
        satellites.append(
            SatelliteConfig(
                orbit_id=0,
                sat_id=i,
                semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                eccentricity=0.001,
                inclination=98.0,
                raan=0.0,
                arg_perigee=i * 90.0,  # 均匀分布
                epoch=datetime.now()
            )
        )
    
    visible_sats = station.get_visible_satellites(satellites)
    print(f"\n可见卫星数: {len(visible_sats)}")
    
    # 验证可见卫星数不超过最大限制
    assert len(visible_sats) <= station.config.max_satellites 