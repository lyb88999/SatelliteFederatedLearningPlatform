import pytest
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator

def test_ground_station_visibility():
    """测试地面站可见性"""
    # 创建轨道计算器（关闭调试模式）
    calculator = OrbitCalculator(debug_mode=False)
    earth_radius = calculator.earth_radius
    
    # 创建地面站（挪威特罗姆瑟）
    ground_station = GroundStationConfig(
        station_id="Tromso",
        latitude=69.6492,
        longitude=18.9553,
        max_range=2500.0,  # 降低通信距离要求
        min_elevation=5.0,
        max_satellites=4
    )
    
    # 创建一颗测试卫星
    satellite = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,  # 正确设置轨道高度
        eccentricity=0.0,
        inclination=98.0,  # 略微调整倾角
        raan=45.0,  # 调整升交点赤经
        arg_perigee=90.0,  # 调整近地点幅角
        epoch=datetime(2024, 1, 1, 0, 0, 0)
    )
    
    # 测试6小时可见性
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    duration = timedelta(hours=6)  # 增加测试时间
    interval = timedelta(minutes=10)  # 增加采样间隔
    
    print("\n=== 卫星和地面站信息 ===")
    print(f"地面站: {ground_station.station_id} ({ground_station.latitude}°N, {ground_station.longitude}°E)")
    print(f"卫星轨道: 高度 {satellite.semi_major_axis - earth_radius:.1f}km, 倾角 {satellite.inclination}°")
    
    print("\n=== 可见性记录 ===")
    print("时间     距离(km)  仰角(°)  可见性")
    print("-" * 40)
    
    current_time = start_time
    visible_count = 0
    total_samples = 0
    
    while current_time < start_time + duration:
        # 计算卫星位置和可见性
        sat_pos = calculator.calculate_satellite_position(satellite, current_time)
        station_pos = calculator.calculate_ground_station_position(ground_station)
        distance = np.linalg.norm(np.array(sat_pos) - np.array(station_pos))
        elevation = calculator._calculate_elevation(np.array(sat_pos), np.array(station_pos))
        is_visible = calculator.check_satellite_visibility(satellite, ground_station, current_time)
        
        print(f"{current_time.strftime('%H:%M')}  {distance:7.0f}  {elevation:7.1f}  {'是' if is_visible else '否'}")
        
        visible_count += 1 if is_visible else 0
        total_samples += 1
        current_time += interval
    
    print("\n=== 统计信息 ===")
    print(f"可见时间: {visible_count}/{total_samples} ({visible_count/total_samples*100:.1f}%)")
    
    # 确保至少有一次可见
    assert visible_count > 0, "卫星在测试期间应该至少有一次可见"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 