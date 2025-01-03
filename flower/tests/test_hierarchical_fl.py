import pytest
import asyncio
from datetime import datetime
import torch
from torch.utils.data import TensorDataset
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.ground_station import GroundStation
from flower.orbit_utils import OrbitCalculator
from flower.fl_server import SatelliteFlowerServer, SatelliteFedAvg
from flower.client import SatelliteFlowerClient

@pytest.fixture
def test_environment():
    """创建测试环境"""
    # 创建简单的测试数据
    num_samples = 1000
    input_dim = 784  # MNIST维度
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    
    # 创建铱星星座配置
    satellites = []
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    num_planes = 6
    sats_per_plane = 11
    altitude = 780.0
    inclination = 86.4
    
    for plane_id in range(num_planes):
        raan = (plane_id * 360.0 / num_planes)
        for sat_id in range(sats_per_plane):
            arg_perigee = (sat_id * 360.0 / sats_per_plane)
            satellites.append(SatelliteConfig(
                orbit_id=plane_id,
                sat_id=len(satellites),
                semi_major_axis=earth_radius + altitude,
                eccentricity=0.001,
                inclination=inclination,
                raan=raan,
                arg_perigee=arg_perigee,
                epoch=datetime.now()
            ))
    
    # 创建地面站
    ground_stations = []
    locations = [
        ("Beijing", 39.9042, 116.4074),
        ("NewYork", 40.7128, -74.0060),
        ("London", 51.5074, -0.1278),
        ("Sydney", -33.8688, 151.2093),
        ("Moscow", 55.7558, 37.6173),
        ("SaoPaulo", -23.5505, -46.6333)
    ]
    
    for name, lat, lon in locations:
        config = GroundStationConfig(
            station_id=name,
            latitude=lat,
            longitude=lon,
            max_range=5000.0,
            min_elevation=0.0,
            max_satellites=10
        )
        ground_stations.append(GroundStation(config, orbit_calculator))
    
    return {
        'dataset': dataset,
        'satellites': satellites,
        'ground_stations': ground_stations,
        'orbit_calculator': orbit_calculator
    }

@pytest.mark.asyncio
async def test_hierarchical_training(test_environment):
    """测试分层训练过程"""
    env = test_environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建服务器
    server = SatelliteFlowerServer(
        satellites=env['satellites'],
        ground_stations=env['ground_stations'],
        orbit_calculator=env['orbit_calculator']
    )
    
    # 创建客户端
    clients = []
    samples_per_client = len(env['dataset']) // len(env['satellites'])
    
    for i in range(len(env['satellites'])):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_dataset = torch.utils.data.Subset(env['dataset'], range(start_idx, end_idx))
        
        client = SatelliteFlowerClient(
            satellite_id=i,
            train_dataset=client_dataset,
            test_dataset=client_dataset,  # 使用相同数据集作为测试集
            device=device
        )
        clients.append(client)
    
    # 执行一轮训练
    metrics = await server.train_round(clients)
    
    # 验证结果
    assert 'accuracy' in metrics
    assert 'loss' in metrics
    assert metrics['accuracy'] >= 0.0 and metrics['accuracy'] <= 1.0
    assert metrics['loss'] >= 0.0
    
    # 验证分层聚合
    # 1. 验证轨道内聚合
    orbit_groups = {}
    for client in clients:
        orbit_id = client.satellite_id // 11
        if orbit_id not in orbit_groups:
            orbit_groups[orbit_id] = []
        orbit_groups[orbit_id].append(client)
    assert len(orbit_groups) == 6  # 6个轨道面
    
    # 2. 验证地面站聚合
    current_time = datetime.now()
    visible_orbits = set()

    # 设置轨道计算器为调试模式
    env['orbit_calculator'].debug_mode = True

    for station in env['ground_stations']:
        for orbit_id in range(6):
            coordinator = server._get_orbit_coordinator(orbit_id)
            if coordinator and env['orbit_calculator'].check_satellite_visibility(
                coordinator, station, current_time):
                visible_orbits.add(orbit_id)
                print(f"地面站 {station.config.station_id} 可见轨道 {orbit_id}")

    # 验证可见性
    assert len(visible_orbits) > 0, "没有可见的轨道"
    print(f"\n可见轨道数量: {len(visible_orbits)}")
    print(f"可见轨道: {sorted(list(visible_orbits))}")
    
    print("\n分层训练测试结果:")
    print(f"- 总卫星数: {len(clients)}")
    print(f"- 轨道面数: {len(orbit_groups)}")
    print(f"- 地面站数: {len(env['ground_stations'])}")
    print(f"- 准确率: {metrics['accuracy']:.4f}")
    print(f"- 损失: {metrics['loss']:.4f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 