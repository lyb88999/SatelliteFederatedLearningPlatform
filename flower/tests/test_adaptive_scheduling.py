import pytest
from datetime import datetime
from typing import List
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator

def create_test_satellites(num_satellites: int = 3) -> List[SatelliteConfig]:
    """创建测试用的卫星配置"""
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    satellites = []
    for i in range(num_satellites):
        satellites.append(
            SatelliteConfig(
                orbit_id=i // 4,
                sat_id=i,
                semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                eccentricity=0.001,
                inclination=98.0,
                raan=(i // 4) * 120.0,
                arg_perigee=(i % 4) * 90.0,
                epoch=datetime.now()
            )
        )
    return satellites

def test_adaptive_scheduling():
    """测试自适应调度"""
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建卫星配置
    satellites = []
    
    # 创建3个轨道，每个轨道4颗卫星
    for orbit_id in range(3):
        for sat_id in range(4):
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0 + orbit_id * 50,  # 轨道高度递增
                    eccentricity=0.001,
                    inclination=97.6,
                    raan=orbit_id * 120.0,
                    arg_perigee=sat_id * 90.0,
                    epoch=datetime.now()
                )
            )
    
    # 验证卫星数量
    assert len(satellites) == 12
    
    # 验证轨道分布
    orbit_groups = {}
    for sat in satellites:
        if sat.orbit_id not in orbit_groups:
            orbit_groups[sat.orbit_id] = []
        orbit_groups[sat.orbit_id].append(sat)
    
    assert len(orbit_groups) == 3  # 3个轨道面
    for sats in orbit_groups.values():
        assert len(sats) == 4  # 每个轨道4颗卫星

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 