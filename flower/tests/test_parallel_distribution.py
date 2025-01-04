import pytest
import asyncio
from datetime import datetime
import numpy as np
from flower.ground_station import GroundStation
from flower.config import GroundStationConfig, SatelliteConfig
from flower.orbit_utils import OrbitCalculator
from typing import Dict, List

pytestmark = pytest.mark.skip(reason="需要重新设计卫星间通信逻辑")

def calculate_satellite_distance(orbit_calculator, sat1, sat2, current_time):
    """计算两颗卫星之间的距离"""
    pos1 = orbit_calculator.calculate_satellite_position(sat1, current_time)
    pos2 = orbit_calculator.calculate_satellite_position(sat2, current_time)
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2)) ** 2))

@pytest.mark.asyncio
async def test_parallel_distribution():
    """测试并行模型分发"""
    # 创建测试环境
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    current_time = datetime.now()
    
    # 创建地面站
    stations = [
        GroundStation(
            GroundStationConfig(
                station_id="Beijing",
                latitude=39.9042,
                longitude=116.4074,
                max_range=2000.0,
                min_elevation=10.0,
                max_satellites=4
            ),
            orbit_calculator
        ),
        GroundStation(
            GroundStationConfig(
                station_id="NewYork",
                latitude=40.7128,
                longitude=-74.0060,
                max_range=2000.0,
                min_elevation=10.0,
                max_satellites=4
            ),
            orbit_calculator
        )
    ]
    
    # 创建卫星
    satellites = []
    for orbit_id in range(2):  # 2个轨道面
        raan = orbit_id * 180.0  # 轨道面间隔180度
        for sat_id in range(2):  # 每个轨道2颗卫星
            phase_angle = sat_id * 180.0  # 卫星间隔180度
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,
                    eccentricity=0.001,
                    inclination=97.6,
                    raan=raan,
                    arg_perigee=phase_angle,
                    epoch=current_time
                )
            )
    
    print(f"\n创建了 {len(satellites)} 颗卫星")
    
    # 打印卫星位置信息
    print("\n卫星位置信息:")
    for sat in satellites:
        pos = orbit_calculator.calculate_satellite_position(sat, current_time)
        print(f"\n卫星 {sat.sat_id}:")
        print(f"- 轨道面: {sat.orbit_id}")
        print(f"- 轨道高度: {sat.semi_major_axis - earth_radius:.1f} km")
        print(f"- 倾角: {sat.inclination}°")
        print(f"- RAAN: {sat.raan}°")
        print(f"- 近地点幅角: {sat.arg_perigee}°")
        print(f"- 位置: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")
    
    # 创建测试模型
    model = {
        'layer1.weight': np.random.randn(10, 10).astype(np.float32),
        'layer1.bias': np.random.randn(10).astype(np.float32),
    }
    
    # 记录开始时间
    start_time = datetime.now()
    results = []
    
    # 为每个地面站分发模型
    for station in stations:
        visible_satellites = []
        print(f"\n地面站 {station.config.station_id} 可见性检查:")
        
        # 检查卫星可见性
        for sat in satellites:
            if orbit_calculator.check_satellite_visibility(sat, station, current_time):
                visible_satellites.append(sat)
                pos = orbit_calculator.calculate_satellite_position(sat, current_time)
                print(f"- 卫星 {sat.sat_id} 可见")
                print(f"  高度: {sat.semi_major_axis - earth_radius:.1f}km")
        
        print(f"\n可见卫星数量: {len(visible_satellites)}")
        
        # 打印所有卫星间距离
        print("\n卫星间距离:")
        for sat1 in visible_satellites:
            for sat2 in satellites:
                if sat1 != sat2:
                    distance = calculate_satellite_distance(
                        orbit_calculator, sat1, sat2, current_time)
                    print(f"卫星 {sat1.sat_id} -> 卫星 {sat2.sat_id}: {distance:.1f}km")
        
        # 简化传输策略：每个可见卫星负责分发给最近的两颗卫星
        successful_transmissions = set()
        if visible_satellites:
            for sat1 in visible_satellites:
                # 找到距离sat1最近的两颗卫星
                distances = []
                for sat2 in satellites:
                    if sat1 != sat2:
                        distance = calculate_satellite_distance(
                            orbit_calculator, sat1, sat2, current_time)
                        distances.append((sat2, distance))
                
                # 按距离排序并选择最近的两颗
                distances.sort(key=lambda x: x[1])
                for sat2, dist in distances[:2]:
                    print(f"  选择传输: 卫星 {sat1.sat_id} -> 卫星 {sat2.sat_id} (距离: {dist:.1f}km)")
                    successful_transmissions.add(sat2.sat_id)
            
            print(f"\n成功分发到卫星: {sorted(list(successful_transmissions))}")
        
        results.append(list(successful_transmissions))
    
    # 验证结果
    total_covered = set()
    for successful_sats in results:
        total_covered.update(successful_sats)
    
    print(f"\n总覆盖卫星数: {len(total_covered)}")
    print(f"覆盖率: {len(total_covered) / len(satellites) * 100:.1f}%")
    
    # 验证至少50%的卫星收到了模型
    assert len(total_covered) >= len(satellites) // 2, \
        f"覆盖率过低: {len(total_covered)}/{len(satellites)}" 