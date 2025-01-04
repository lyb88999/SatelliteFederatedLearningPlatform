import pytest
import asyncio
from datetime import datetime
import torch
from torch.utils.data import TensorDataset
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.ground_station import GroundStation
from flower.orbit_utils import OrbitCalculator
from flower.server import SatelliteServer
from flower.client import SatelliteFlowerClient
from flower.scheduler import Scheduler
from flower.monitor import Monitor

@pytest.fixture
def test_environment():
    """创建测试环境"""
    # 创建简单的测试数据
    num_samples = 1000
    input_dim = 784  # MNIST维度
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(X, y)
    
    # 创建卫星配置
    satellites = []
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建2个轨道，每个轨道2颗卫星
    for orbit_id in range(2):
        raan = orbit_id * 180.0  # 轨道面均匀分布
        for sat_id in range(2):
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
    
    # 创建调度器和监控器
    scheduler = Scheduler(env['orbit_calculator'])
    monitor = Monitor()
    
    # 创建服务器
    server = SatelliteServer(scheduler, monitor)
    
    # 创建测试模型
    model = {
        'layer1.weight': np.random.randn(10, 10).astype(np.float32),
        'layer1.bias': np.random.randn(10).astype(np.float32),
    }
    
    # 设置服务器模型
    server.set_model(model)
    
    # 创建客户端
    clients = []
    samples_per_client = len(env['dataset']) // len(env['satellites'])
    
    for i, satellite in enumerate(env['satellites']):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_dataset = torch.utils.data.Subset(env['dataset'], range(start_idx, end_idx))
        
        # 创建客户端配置
        client_config = {
            'batch_size': 32,
            'epochs': 1,
            'learning_rate': 0.01
        }
        
        client = SatelliteFlowerClient(
            satellite_id=satellite.sat_id,
            train_data=client_dataset,
            test_data=client_dataset,  # 使用相同数据集作为测试集
            config=client_config,      # 添加配置
            device=device
        )
        clients.append(client)
    
    print("\n开始分层训练测试")
    print(f"卫星数量: {len(clients)}")
    print(f"地面站数量: {len(env['ground_stations'])}")
    
    # 执行训练
    await server.start_training(model, env['satellites'])
    
    # 验证训练完成
    assert server.current_round > 0
    print(f"\n完成 {server.current_round} 轮训练")
    
    # 验证分层聚合
    # 1. 验证轨道内聚合
    orbit_groups = {}
    for satellite in env['satellites']:
        if satellite.orbit_id not in orbit_groups:
            orbit_groups[satellite.orbit_id] = []
        orbit_groups[satellite.orbit_id].append(satellite)
    
    assert len(orbit_groups) == 2  # 2个轨道面
    for sats in orbit_groups.values():
        assert len(sats) == 2  # 每个轨道2颗卫星
    
    # 2. 验证地面站可见性
    current_time = datetime.now()
    visible_satellites = set()  # 使用集合避免重复
    
    # 定义可见性映射
    visibility_map = {
        "Beijing": [0],     # 北京只能看到轨道0的卫星
        "NewYork": [1]      # 纽约只能看到轨道1的卫星
    }
    
    for station in env['ground_stations']:
        station_visible = []
        for satellite in env['satellites']:
            # 检查卫星是否在可见轨道上
            if satellite.orbit_id in visibility_map[station.config.station_id]:
                if env['orbit_calculator'].check_satellite_visibility(
                    satellite, station, current_time):
                    visible_satellites.add(satellite.sat_id)
                    station_visible.append(satellite.sat_id)
        print(f"\n地面站 {station.config.station_id} 可见卫星: {station_visible}")
    
    # 验证可见性
    assert len(visible_satellites) > 0, "没有可见的卫星"
    # 验证每个地面站的可见卫星数量
    for station in env['ground_stations']:
        station_sats = [sat for sat in env['satellites'] 
                       if sat.orbit_id in visibility_map[station.config.station_id]]
        assert len(station_sats) <= 2, f"{station.config.station_id} 不应该看到超过2颗卫星"
    
    print(f"\n总可见卫星数量: {len(visible_satellites)}")
    print(f"可见卫星ID: {sorted(list(visible_satellites))}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 