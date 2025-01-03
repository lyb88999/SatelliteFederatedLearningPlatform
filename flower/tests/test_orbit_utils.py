import pytest
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig
from flower.orbit_utils import OrbitCalculator

def test_satellite_position():
    """测试卫星位置计算"""
    calculator = OrbitCalculator(debug_mode=False)
    earth_radius = calculator.earth_radius
    
    sat_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    pos = calculator.calculate_satellite_position(sat_config, datetime.now())
    assert all(isinstance(x, float) for x in pos)

def test_satellite_distance():
    """测试卫星间距离计算"""
    calculator = OrbitCalculator(debug_mode=False)
    current_time = datetime.now()
    
    sat1_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=calculator.earth_radius + 550.0,
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    sat2_config = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=calculator.earth_radius + 550.0,
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=90.0,  # 90度相位差
        epoch=datetime.now()
    )
    
    distance = calculator.calculate_satellite_distance(sat1_config, sat2_config, current_time)
    assert distance > 0

def test_communication_window():
    """测试通信窗口检查"""
    calculator = OrbitCalculator(debug_mode=False)
    current_time = datetime.now()
    
    sat1_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=calculator.earth_radius + 550.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        max_communication_distance=1000.0,  # 1000km通信距离
        epoch=datetime.now()
    )
    
    sat2_config = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=calculator.earth_radius + 550.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=10.0,  # 10度相位差
        epoch=datetime.now()
    )
    
    has_window = calculator.check_satellite_visibility(sat1_config, sat2_config, current_time)
    distance = calculator.calculate_satellite_distance(sat1_config, sat2_config, current_time)
    print(f"\n通信窗口测试结果: {has_window}, 类型: {type(has_window)}")
    print(f"卫星间距离: {distance:.2f} km")
    print(f"最大通信距离: {sat1_config.max_communication_distance} km")
    
    assert isinstance(has_window, bool)
    assert distance > 0

def test_earth_obstruction():
    """测试地球遮挡检查"""
    calculator = OrbitCalculator(debug_mode=False)
    earth_radius = calculator.earth_radius
    
    # 创建两个相对的卫星
    sat1_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    sat2_config = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=earth_radius + 550.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=180.0,  # 对面的卫星
        epoch=datetime.now()
    )
    
    # 检查地球遮挡
    current_time = datetime.now()
    pos1 = calculator.calculate_satellite_position(sat1_config, current_time)
    pos2 = calculator.calculate_satellite_position(sat2_config, current_time)
    is_obstructed = calculator._check_earth_obstruction(pos1, pos2)
    
    assert isinstance(is_obstructed, bool), "地球遮挡检查应该返回布尔值"
    assert is_obstructed, "对面的卫星应该被地球遮挡"

def test_orbit_propagation():
    """测试轨道传播"""
    calculator = OrbitCalculator(debug_mode=False)
    
    # 创建一个测试卫星配置，使用当前时间作为历元
    epoch = datetime.now()
    sat_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=7000.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        eccentricity=0.0,  # 圆轨道更容易验证
        epoch=epoch  # 设置历元时间
    )
    
    # 计算轨道周期
    orbital_period = calculator.calculate_orbital_period(sat_config.semi_major_axis)
    print(f"\n轨道周期: {orbital_period/60:.2f} 分钟")
    
    # 记录一个轨道周期内的位置
    positions = []
    times = []
    for i in range(5):  # 记录5个点，包括起点和终点
        t = epoch + timedelta(seconds=i*orbital_period/4)
        times.append(t)
        pos = calculator.calculate_satellite_position(sat_config, t)
        positions.append(pos)
        print(f"时间点 {i}: {pos}")
    
    # 验证轨道是否闭合
    start_pos = np.array(positions[0])
    end_pos = np.array(positions[-1])
    
    # 计算绝对误差和相对误差
    abs_error = np.abs(end_pos - start_pos)
    print(f"\n起始位置: {start_pos}")
    print(f"终止位置: {end_pos}")
    print(f"绝对误差: {abs_error}")
    
    # 使用轨道半径作为参考尺度
    orbit_radius = sat_config.semi_major_axis
    normalized_error = abs_error / orbit_radius
    print(f"归一化误差: {normalized_error}")
    
    # 验证轨道闭合性
    assert np.all(normalized_error < 1e-6), "轨道未闭合，误差过大"
    
    # 验证轨道半径
    for pos in positions:
        radius = np.sqrt(np.sum(np.array(pos)**2))
        radius_error = abs(radius - orbit_radius) / orbit_radius
        print(f"轨道半径: {radius:.2f} km, 相对误差: {radius_error:.2e}")
        assert radius_error < 1e-6, "轨道半径误差过大"

def test_real_orbit_calculations():
    """测试实际轨道计算"""
    calculator = OrbitCalculator(debug_mode=False)
    current_time = datetime.now()
    
    # 创建一个LEO卫星配置
    sat_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=7000.0,  # LEO轨道
        eccentricity=0.001,      # 近圆轨道
        inclination=98.0,        # 太阳同步轨道
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    # 1. 测试位置计算
    pos = calculator.calculate_satellite_position(sat_config, datetime.now())
    print(f"\n卫星位置: {pos}")
    assert all(isinstance(x, float) for x in pos)
    
    # 2. 测试轨道周期
    period = calculator.calculate_orbital_period(sat_config.semi_major_axis)
    print(f"轨道周期: {period/60:.2f} 分钟")
    assert 90 < period/60 < 100  # LEO轨道周期约为90-100分钟
    
    # 3. 测试地面轨迹
    track = calculator.calculate_ground_track(sat_config, datetime.now(), 1.0)
    print(f"地面轨迹点数: {len(track)}")
    assert len(track) > 0
    assert all(-180 <= lon <= 180 and -90 <= lat <= 90 for lon, lat in track)
    
    # 4. 测试通信窗口
    sat2_config = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=7000.0,
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=10.0,  # 相位差10度
        epoch=datetime.now()
    )
    
    has_window = calculator.check_communication_window(sat_config, sat2_config)
    distance = calculator.calculate_satellite_distance(sat_config, sat2_config, current_time)
    print(f"卫星间距离: {distance:.2f} km")
    print(f"通信窗口: {has_window}")
    
    # 验证结果
    assert isinstance(has_window, bool)
    assert distance > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 