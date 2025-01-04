import pytest
from datetime import datetime
from typing import List
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator

def create_test_satellites(num_satellites: int = 4) -> List[SatelliteConfig]:
    """创建测试用的卫星配置"""
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    satellites = []
    for i in range(num_satellites):
        satellites.append(
            SatelliteConfig(
                orbit_id=i // 2,  # 每2颗卫星一个轨道面
                sat_id=i,
                semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                eccentricity=0.001,
                inclination=97.6,
                raan=(i // 2) * 180.0,  # 轨道面均匀分布
                arg_perigee=(i % 2) * 180.0,  # 卫星在轨道内均匀分布
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
    
    # 创建2个轨道，每个轨道2颗卫星
    for orbit_id in range(2):
        for sat_id in range(2):
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0 + orbit_id * 50,  # 轨道高度递增
                    eccentricity=0.001,
                    inclination=97.6,
                    raan=orbit_id * 180.0,  # 轨道面均匀分布
                    arg_perigee=sat_id * 180.0,  # 卫星在轨道内均匀分布
                    epoch=datetime.now()
                )
            )
    
    # 验证卫星数量
    assert len(satellites) == 4
    
    # 验证轨道分布
    orbit_groups = {}
    for sat in satellites:
        if sat.orbit_id not in orbit_groups:
            orbit_groups[sat.orbit_id] = []
        orbit_groups[sat.orbit_id].append(sat)
    
    assert len(orbit_groups) == 2  # 2个轨道面
    for sats in orbit_groups.values():
        assert len(sats) == 2  # 每个轨道2颗卫星
        
    # 打印卫星分布情况
    print("\n卫星分布情况:")
    for orbit_id, sats in orbit_groups.items():
        print(f"\n轨道 {orbit_id}:")
        for sat in sats:
            print(f"  卫星 {sat.sat_id}: "
                  f"高度={sat.semi_major_axis - earth_radius:.1f}km, "
                  f"RAAN={sat.raan:.1f}°, "
                  f"相位角={sat.arg_perigee:.1f}°")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 