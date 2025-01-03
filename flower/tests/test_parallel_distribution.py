import pytest
import asyncio
from datetime import datetime
import numpy as np
from flower.ground_station import GroundStation
from flower.config import GroundStationConfig, SatelliteConfig
from flower.orbit_utils import OrbitCalculator
from typing import Dict, List

def calculate_transmission_time(calculator, sat1: SatelliteConfig, sat2: SatelliteConfig, 
                              current_time: datetime, speed_of_light: float = 299792.458) -> float:
    """计算传输时间（秒）
    
    Args:
        calculator: 轨道计算器
        sat1: 发送卫星
        sat2: 接收卫星
        current_time: 当前时间
        speed_of_light: 光速（km/s），默认值约为 299,792.458 km/s
        
    Returns:
        float: 传输时间（秒）
    """
    try:
        # 计算卫星间距离（km）
        distance = calculator.calculate_satellite_distance(sat1, sat2, current_time)
        # 计算传输时间（秒）= 距离（km）/ 光速（km/s）
        return distance / speed_of_light
    except Exception as e:
        print(f"传输时间计算错误: {str(e)}")
        return float('inf')  # 如果计算失败，返回无穷大表示无法传输

@pytest.mark.asyncio
async def test_parallel_distribution():
    """测试并行模型分发"""
    # 创建测试环境
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius  # 获取地球半径
    
    # 创建地面站
    stations = [
        GroundStation(
            GroundStationConfig(
                station_id="Beijing",
                latitude=39.9042,
                longitude=116.4074,
                max_range=1000.0,    # 降低最大通信距离到1000km
                min_elevation=10.0,   # 提高最小仰角到10度
                max_satellites=4
            ),
            orbit_calculator
        ),
        GroundStation(
            GroundStationConfig(
                station_id="Shanghai",
                latitude=31.2304,
                longitude=121.4737,
                max_range=1000.0,
                min_elevation=10.0,
                max_satellites=4
            ),
            orbit_calculator
        )
    ]
    
    # 创建卫星
    satellites = []
    for orbit_id in range(3):  # 3个轨道面
        raan = orbit_id * 120.0  # 轨道面均匀分布
        for sat_id in range(4):  # 每个轨道面4颗卫星
            # 计算初始相位角（在轨道平面内的位置）
            phase_angle = sat_id * 90.0 + raan  # 考虑轨道面的旋转
            
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,  # 550km轨道高度（如Starlink）
                    eccentricity=0.001,      # 近圆轨道
                    inclination=98.0,        # 太阳同步轨道
                    raan=raan,              # 轨道面的方向
                    arg_perigee=phase_angle, # 卫星在轨道内的位置
                    epoch=datetime.now()
                )
            )
    
    print(f"\n创建了 {len(satellites)} 颗卫星")
    print("轨道参数示例:")
    for i in range(min(3, len(satellites))):
        sat = satellites[i]
        print(f"\n卫星 {sat.sat_id}:")
        print(f"- 轨道面: {sat.orbit_id}")
        print(f"- 轨道高度: {sat.semi_major_axis - earth_radius:.1f} km")
        print(f"- 倾角: {sat.inclination}°")
        print(f"- 升交点赤经: {sat.raan}°")
        print(f"- 近地点幅角: {sat.arg_perigee}°")
    
    # 创建测试模型
    model = {
        'layer1.weight': np.random.randn(100, 100).astype(np.float32),
        'layer1.bias': np.random.randn(100).astype(np.float32),
        'layer2.weight': np.random.randn(100, 10).astype(np.float32),
        'layer2.bias': np.random.randn(10).astype(np.float32)
    }
    
    # 打印模型大小
    total_size = sum(param.nbytes for param in model.values())
    print(f"\n模型大小: {total_size/1e6:.2f} MB")
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 存储每个地面站的分发结果
    results = []
    
    # 在检查可见性和计算传输时间时
    current_time = datetime.now()
    
    for station in stations:
        visible_satellites = []
        for sat in satellites:
            if orbit_calculator.check_satellite_visibility(sat, station, current_time):
                visible_satellites.append(sat)
                
        print(f"\n地面站 {station.config.station_id} 可见性检查结果:")
        print(f"检查的卫星总数: {len(satellites)}")
        print(f"可见卫星数: {len(visible_satellites)}")
        
        # 计算传输时间
        successful_transmissions = []
        if visible_satellites:
            transmission_times = []
            for sat1 in visible_satellites:
                for sat2 in satellites:
                    if sat1 != sat2 and sat2.sat_id not in successful_transmissions:
                        time = calculate_transmission_time(orbit_calculator, sat1, sat2, current_time)
                        transmission_times.append((sat1, sat2, time))
                        
            # 按传输时间排序
            transmission_times.sort(key=lambda x: x[2])
            
            # 选择传输时间最短的路径，每个卫星只选一次
            used_satellites = set()  # 用于跟踪已经使用的卫星
            for sat1, sat2, time in transmission_times:
                if (sat2.sat_id not in successful_transmissions and 
                    sat1.sat_id not in used_satellites and 
                    sat2.sat_id not in used_satellites):
                    successful_transmissions.append(sat2.sat_id)
                    used_satellites.add(sat1.sat_id)
                    used_satellites.add(sat2.sat_id)
                    
            print(f"\n地面站 {station.config.station_id}:")
            print(f"- 成功分发到卫星: {sorted(successful_transmissions)}")
            
            total_time = sum(t[2] for t in transmission_times[:len(successful_transmissions)])
            print(f"\n总分发时间: {total_time:.2f} 秒")
            
        results.append(successful_transmissions)
    
    # 记录结束时间
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 验证结果
    for station_idx, successful_sats in enumerate(results):
        print(f"\n地面站 {stations[station_idx].config.station_id} 最终结果:")
        print(f"- 成功分发到卫星: {sorted(successful_sats)}")
    
    print(f"\n总分发时间: {total_time:.2f} 秒")
    
    # 验证结果合理性
    for successful_sats in results:
        # 验证至少有一些卫星接收到了模型
        assert len(successful_sats) > 0, "没有卫星接收到模型"
        # 验证没有重复的卫星
        assert len(successful_sats) == len(set(successful_sats)), "存在重复的卫星"
        # 验证卫星ID在合理范围内
        assert all(0 <= sat_id < len(satellites) for sat_id in successful_sats), "卫星ID超出范围"
        # 验证分发数量合理
        assert len(successful_sats) <= len(satellites) // 2, "分发数量过多"
    
    # 验证分发时间合理
    assert total_time < len(satellites), "分发时间过长" 