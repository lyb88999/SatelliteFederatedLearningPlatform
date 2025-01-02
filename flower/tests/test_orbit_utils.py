import pytest
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig
from flower.orbit_utils import OrbitCalculator

def test_satellite_position():
    """测试卫星位置计算"""
    calculator = OrbitCalculator(debug_mode=False)
    
    # 创建一个测试卫星配置
    sat_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=7000.0,  # LEO轨道
        eccentricity=0.0,        # 圆轨道
        inclination=98.0,        # 太阳同步轨道
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    # 测试位置计算
    time = datetime.now()
    pos = calculator.calculate_satellite_position(sat_config, time)
    
    # 验证位置是否合理
    assert len(pos) == 3  # 三维坐标
    assert all(isinstance(x, float) for x in pos)  # 数值类型正确
    
    # 验证轨道高度
    distance_from_center = np.sqrt(sum(x*x for x in pos))
    assert abs(distance_from_center - sat_config.semi_major_axis) < 1.0  # 误差小于1km

def test_satellite_distance():
    """测试卫星间距离计算"""
    calculator = OrbitCalculator(debug_mode=False)
    
    # 创建两个测试卫星配置
    sat1_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=7000.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0
    )
    
    sat2_config = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=7000.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=90.0  # 相位差90度
    )
    
    # 计算距离
    distance = calculator.calculate_satellite_distance(sat1_config, sat2_config)
    
    # 验证距离是否合理
    assert isinstance(distance, float)
    assert distance > 0
    # 对于相位差90度的同轨道卫星，距离应该约为 2*R*sin(45°)
    expected_distance = 2 * 7000.0 * np.sin(np.pi/4)
    assert abs(distance - expected_distance) < 100  # 允许100km的误差

def test_communication_window():
    """测试通信窗口检查"""
    calculator = OrbitCalculator(debug_mode=False)
    
    # 创建两个相邻的测试卫星配置
    sat1_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=7000.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        max_communication_distance=2000.0
    )
    
    sat2_config = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=7000.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=10.0
    )
    
    # 测试通信窗口
    has_window = calculator.check_communication_window(sat1_config, sat2_config)
    print(f"\n通信窗口测试结果: {has_window}, 类型: {type(has_window)}")
    
    # 检查返回值类型
    assert isinstance(has_window, bool), f"返回值类型错误: {type(has_window)}"
    
    # 计算实际距离
    distance = calculator.calculate_satellite_distance(sat1_config, sat2_config)
    print(f"卫星间距离: {distance:.2f} km")
    print(f"最大通信距离: {sat1_config.max_communication_distance} km")
    
    # 相位差10度的卫星应该在通信范围内
    assert has_window == True, f"相位差10度的卫星应该在通信范围内，但返回了 {has_window}"
    
    # 测试地球遮挡
    sat2_config.arg_perigee = 180.0
    has_window = calculator.check_communication_window(sat1_config, sat2_config)
    assert has_window == False, "地球遮挡时应该返回 False"

def test_earth_obstruction():
    """测试地球遮挡检查"""
    calculator = OrbitCalculator(debug_mode=False)
    
    # 测试明显被地球遮挡的情况
    pos1 = (7000.0, 0.0, 0.0)  # 卫星1在x轴正方向
    pos2 = (-7000.0, 0.0, 0.0)  # 卫星2在x轴负方向
    
    is_obstructed = calculator._check_earth_obstruction(pos1, pos2)
    assert is_obstructed == True
    
    # 测试明显不被遮挡的情况
    pos2 = (7000.0, 1000.0, 0.0)  # 卫星2在附近
    is_obstructed = calculator._check_earth_obstruction(pos1, pos2)
    assert is_obstructed == False

def test_orbit_propagation():
    """测试轨道传播"""
    calculator = OrbitCalculator(debug_mode=False)
    
    sat_config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=7000.0,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0
    )
    
    # 计算一个轨道周期内的位置
    time = datetime.now()
    orbital_period = 2 * np.pi * np.sqrt(sat_config.semi_major_axis**3 / 398600.4418)  # 轨道周期(s)
    
    positions = []
    for i in range(4):  # 记录4个点
        t = time + timedelta(seconds=i*orbital_period/4)
        pos = calculator.calculate_satellite_position(sat_config, t)
        positions.append(pos)
    
    # 验证轨道是否闭合
    start_pos = np.array(positions[0])
    end_pos = np.array(positions[-1])
    assert np.allclose(start_pos, end_pos, rtol=0.1)  # 允许10%的相对误差

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 