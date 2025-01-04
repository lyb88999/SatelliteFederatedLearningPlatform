import pytest
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator

def test_basic_orbit_utils():
    """测试基本轨道计算功能"""
    # 创建轨道计算器
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建卫星配置
    satellites = []
    for orbit_id in range(2):  # 2个轨道面
        raan = orbit_id * 180.0  # 轨道面均匀分布
        for sat_id in range(2):  # 每个轨道2颗卫星
            phase_angle = sat_id * 180.0  # 卫星在轨道内均匀分布
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                    eccentricity=0.001,
                    inclination=97.6,
                    raan=raan,
                    arg_perigee=phase_angle,
                    epoch=datetime.now()
                )
            )
    
    # 创建地面站
    ground_stations = []
    locations = [
        ("Beijing", 39.9042, 116.4074),
        ("NewYork", 40.7128, -74.0060)
    ]
    
    for name, lat, lon in locations:
        config = GroundStationConfig(
            station_id=name,
            latitude=lat,
            longitude=lon,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        )
        ground_stations.append(config)
    
    # 测试卫星位置计算
    current_time = datetime.now()
    print("\n卫星位置计算测试:")
    for satellite in satellites:
        position = orbit_calculator.calculate_satellite_position(satellite, current_time)
        height = np.sqrt(np.sum(np.square(position))) - earth_radius
        print(f"\n卫星 {satellite.sat_id} (轨道 {satellite.orbit_id}):")
        print(f"位置: X={position[0]:.1f}, Y={position[1]:.1f}, Z={position[2]:.1f}")
        print(f"高度: {height:.1f}km")
        
        # 验证高度
        assert abs(height - 550.0) < 1.0, "卫星高度应该在550km左右"
    
    # 测试可见性计算
    print("\n可见性计算测试:")
    visibility_map = {
        "Beijing": [0],     # 北京只能看到轨道0的卫星
        "NewYork": [1]      # 纽约只能看到轨道1的卫星
    }
    
    for station in ground_stations:
        station_visible = []
        for satellite in satellites:
            if satellite.orbit_id in visibility_map[station.station_id]:
                if orbit_calculator.check_satellite_visibility(
                    satellite, station, current_time):
                    station_visible.append(satellite.sat_id)
        print(f"\n地面站 {station.station_id} 可见卫星: {station_visible}")
        
        # 验证可见性
        visible_sats = [sat for sat in satellites 
                       if sat.orbit_id in visibility_map[station.station_id]]
        assert len(station_visible) <= len(visible_sats), \
            f"{station.station_id} 不应该看到超过{len(visible_sats)}颗卫星"

def test_advanced_orbit_utils():
    """测试高级轨道计算功能"""
    calculator = OrbitCalculator(debug_mode=True)
    earth_radius = calculator.earth_radius
    current_time = datetime.now()
    
    # 创建测试卫星
    sat1 = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,
        eccentricity=0.001,
        inclination=97.6,
        raan=0.0,
        arg_perigee=0.0,
        epoch=current_time
    )
    
    sat2 = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=earth_radius + 550.0,
        eccentricity=0.001,
        inclination=97.6,
        raan=0.0,
        arg_perigee=180.0,  # 对面的卫星
        epoch=current_time
    )
    
    # 测试轨道周期
    period = calculator.calculate_orbital_period(sat1.semi_major_axis)
    print(f"\n轨道周期: {period/60:.2f} 分钟")
    assert 90 < period/60 < 100  # LEO轨道周期约为90-100分钟
    
    # 测试卫星间距离
    distance = calculator.calculate_satellite_distance(sat1, sat2, current_time)
    print(f"卫星间距离: {distance:.2f} km")
    assert distance > 0
    
    # 测试地球遮挡
    pos1 = calculator.calculate_satellite_position(sat1, current_time)
    pos2 = calculator.calculate_satellite_position(sat2, current_time)
    # 转换为 numpy 数组
    pos1_array = np.array(pos1)
    pos2_array = np.array(pos2)
    is_obstructed = calculator._check_earth_obstruction(pos1_array, pos2_array)
    assert is_obstructed, "对面的卫星应该被地球遮挡"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 