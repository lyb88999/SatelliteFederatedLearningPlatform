import pytest
import torch
from datetime import datetime
import numpy as np
from typing import List, Dict
from flower.config import SatelliteConfig
from flower.orbit_utils import OrbitCalculator
from flower.client import SatelliteFlowerClient
from flower.server import SatelliteServer
from flower.scheduler import Scheduler
from flower.monitor import Monitor

class SimpleTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def large_scale_environment():
    """创建大规模测试环境"""
    # 创建轨道计算器
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建卫星配置
    satellites = []
    for orbit_id in range(2):  # 2个轨道面
        raan = orbit_id * 90.0  # 轨道面间隔90度，而不是180度
        for sat_id in range(2):  # 每个轨道2颗卫星
            phase_angle = sat_id * 180.0  # 同一轨道上的卫星间隔180度
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                    eccentricity=0.001,
                    inclination=97.6,
                    raan=raan,  # 轨道面间隔90度
                    arg_perigee=phase_angle,  # 卫星间隔180度
                    epoch=datetime.now()
                )
            )
    
    # 创建测试数据
    num_samples = 1000
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    
    # 创建客户端
    device = torch.device("cpu")
    clients = []
    samples_per_client = len(dataset) // len(satellites)
    
    for i, satellite in enumerate(satellites):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        
        client_config = {
            'batch_size': 32,
            'epochs': 1,
            'learning_rate': 0.01
        }
        
        client = SatelliteFlowerClient(
            satellite_id=satellite.sat_id,
            train_data=client_dataset,
            test_data=client_dataset,  # 使用相同数据集作为测试集
            config=client_config,
            device=device
        )
        clients.append(client)
    
    # 创建调度器和监控器
    scheduler = Scheduler(orbit_calculator)
    monitor = Monitor()
    
    return {
        'satellites': satellites,
        'clients': clients,
        'scheduler': scheduler,
        'monitor': monitor,
        'orbit_calculator': orbit_calculator
    }

def test_multi_orbit_visibility(large_scale_environment):
    """测试多轨道可见性"""
    env = large_scale_environment
    current_time = datetime.now()
    
    # 验证轨道分布
    orbit_groups = {}
    for sat in env['satellites']:
        if sat.orbit_id not in orbit_groups:
            orbit_groups[sat.orbit_id] = []
        orbit_groups[sat.orbit_id].append(sat)
    
    assert len(orbit_groups) == 2, "应该有2个轨道面"
    for sats in orbit_groups.values():
        assert len(sats) == 2, "每个轨道应该有2颗卫星"
    
    print("\n轨道分布验证通过")
    
    # 验证卫星间距离
    min_dist, max_dist = check_satellite_distances(
        env['satellites'], env['orbit_calculator'], current_time)

@pytest.mark.asyncio
async def test_large_scale_scheduling(large_scale_environment):
    """测试大规模任务调度"""
    env = large_scale_environment
    server = SatelliteServer(env['scheduler'], env['monitor'])
    
    # 创建测试模型
    model = {
        'layer1.weight': np.random.randn(10, 10).astype(np.float32),
        'layer1.bias': np.random.randn(10).astype(np.float32),
    }
    
    # 执行训练
    print("\n开始大规模训练测试")
    print(f"卫星数量: {len(env['satellites'])}")
    
    server.set_model(model)
    await server.start_training(model, env['satellites'])
    
    assert server.current_round > 0, "训练应该完成至少一轮"
    print(f"\n完成 {server.current_round} 轮训练")

def test_coordinator_network(large_scale_environment):
    """测试协调者网络"""
    env = large_scale_environment
    current_time = datetime.now()
    
    # 验证每个轨道内的通信
    for orbit_id in range(2):
        orbit_sats = [sat for sat in env['satellites'] if sat.orbit_id == orbit_id]
        print(f"\n轨道 {orbit_id} 内部通信测试:")
        
        for i, sat1 in enumerate(orbit_sats):
            for sat2 in orbit_sats[i+1:]:
                distance = env['orbit_calculator'].calculate_satellite_distance(
                    sat1, sat2, current_time)
                print(f"卫星 {sat1.sat_id} 到卫星 {sat2.sat_id} 的距离: {distance:.2f}km")
                assert distance > 0, "轨道内卫星间距离应该大于0"

def check_satellite_distances(satellites, orbit_calculator, current_time):
    """检查卫星间距离的合理性"""
    earth_radius = orbit_calculator.earth_radius
    min_distance = float('inf')
    max_distance = 0
    
    # 计算轨道半径和周长
    orbit_radius = earth_radius + 550.0  # 550km轨道高度
    orbit_circumference = 2 * np.pi * orbit_radius
    
    # 检查所有卫星对之间的距离
    for i, sat1 in enumerate(satellites):
        pos1 = np.array(orbit_calculator.calculate_satellite_position(sat1, current_time))  # 转换为 numpy 数组
        for j, sat2 in enumerate(satellites[i+1:], i+1):
            pos2 = np.array(orbit_calculator.calculate_satellite_position(sat2, current_time))  # 转换为 numpy 数组
            
            # 计算欧氏距离
            distance = np.sqrt(np.sum((pos1 - pos2) ** 2))
            min_distance = min(min_distance, distance)
            max_distance = max(max_distance, distance)
    
    print(f"\n卫星距离统计:")
    print(f"最小距离: {min_distance:.2f}km")
    print(f"最大距离: {max_distance:.2f}km")
    print(f"轨道半径: {orbit_radius:.2f}km")
    print(f"轨道周长: {orbit_circumference:.2f}km")
    
    return min_distance, max_distance

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 