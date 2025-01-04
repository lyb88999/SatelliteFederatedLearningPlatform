import pytest
from datetime import datetime
from flower.config import SatelliteConfig, GroundStationConfig
from flower.ground_station import GroundStation
from flower.orbit_utils import OrbitCalculator

def test_ground_station_visibility():
    """测试地面站可见性检查"""
    # 创建轨道计算器（使用调试模式）
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建北京地面站
    station_config = GroundStationConfig(
        station_id="Beijing",
        latitude=39.9042,
        longitude=116.4074,
        max_range=2000.0,    # 2000km通信距离
        min_elevation=10.0,   # 10度最小仰角
        max_satellites=4      # 最多同时连接4颗卫星
    )
    
    station = GroundStation(station_config, orbit_calculator)
    
    # 创建测试卫星 - 分布在不同轨道面
    satellites = []
    for i in range(4):
        satellites.append(
            SatelliteConfig(
                orbit_id=i // 2,  # 每2颗卫星一个轨道面
                sat_id=i,
                semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                eccentricity=0.001,
                inclination=97.6,
                raan=i * 90.0,  # 不同的升交点赤经
                arg_perigee=i * 90.0,  # 均匀分布在轨道上
                epoch=datetime.now()
            )
        )
    
    # 获取可见卫星
    visible_sats = station.get_visible_satellites(satellites)
    print(f"\n地面站: {station_config.station_id}")
    print(f"可见卫星数: {len(visible_sats)}")
    for sat in visible_sats:
        print(f"卫星ID: {sat.sat_id}, 轨道ID: {sat.orbit_id}")
    
    # 验证可见卫星数不超过最大限制
    assert len(visible_sats) <= station_config.max_satellites
    
    # 验证可见性窗口
    current_time = datetime.now()
    for sat in satellites:
        is_visible = orbit_calculator.check_satellite_visibility(
            sat, station_config, current_time
        )
        if is_visible:
            print(f"\n卫星 {sat.sat_id} 当前可见")
            print(f"轨道高度: {sat.semi_major_axis - earth_radius:.1f}km")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 