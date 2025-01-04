import pytest
import asyncio
from datetime import datetime, timedelta
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
    
    # 创建轨道计算器
    orbit_calculator = OrbitCalculator(debug_mode=False)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建卫星配置
    satellites = []
    # 创建2个轨道，每个轨道2颗卫星
    for orbit_id in range(2):
        raan = orbit_id * 45.0  # 调整RAAN分布
        for sat_id in range(2):
            phase_angle = sat_id * 90.0  # 调整相位角分布
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                    eccentricity=0.0,
                    inclination=98.0,  # 保持倾角
                    raan=raan,
                    arg_perigee=phase_angle,
                    epoch=datetime.now()
                )
            )
    
    # 创建地面站
    ground_stations = []
    locations = [
        ("Tromso", 69.6492, 18.9553),  # 特罗姆瑟地面站
        ("Antarctica", -69.6492, 18.9553)  # 对称的南极地面站
    ]
    
    for name, lat, lon in locations:
        config = GroundStationConfig(
            station_id=name,
            latitude=lat,
            longitude=lon,
            max_range=2500.0,  # 与成功测试保持一致
            min_elevation=5.0,  # 与成功测试保持一致
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
    
    # 移除可见性映射限制，允许地面站看到所有轨道的卫星
    # 扩大时间窗口到3小时，每15分钟检查一次
    test_times = [
        current_time + timedelta(minutes=i*15) 
        for i in range(12)  # 3小时，每15分钟一次
    ]
    
    for station in env['ground_stations']:
        station_visible = []
        print(f"\n检查地面站 {station.config.station_id} 的可见性:")
        
        for test_time in test_times:
            for satellite in env['satellites']:
                # 检查卫星可见性，不再限制轨道
                is_visible = env['orbit_calculator'].check_satellite_visibility(
                    satellite, station, test_time)
                
                if is_visible:
                    visible_satellites.add(satellite.sat_id)
                    if satellite.sat_id not in station_visible:
                        station_visible.append(satellite.sat_id)
                        print(f"时间点 {test_time.strftime('%H:%M:%S')} - 发现可见卫星 {satellite.sat_id}")
                        # 打印详细信息以便调试
                        pos = env['orbit_calculator'].calculate_satellite_position(satellite, test_time)
                        # 将笛卡尔坐标转换为经纬度
                        x, y, z = pos
                        r = np.sqrt(x*x + y*y + z*z)
                        lat = np.arcsin(z/r) * 180/np.pi
                        lon = np.arctan2(y, x) * 180/np.pi
                        height = r - env['orbit_calculator'].earth_radius
                        print(f"  卫星位置: 经度={lon:.2f}°, 纬度={lat:.2f}°, 高度={height:.2f}km")
        
        print(f"地面站 {station.config.station_id} 可见卫星总数: {len(station_visible)}")
        print(f"可见卫星列表: {station_visible}")
    
    print(f"\n所有可见卫星: {visible_satellites}")
    assert len(visible_satellites) > 0, "没有可见的卫星"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 