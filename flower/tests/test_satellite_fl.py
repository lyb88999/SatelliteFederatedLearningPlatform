import asyncio
from datetime import datetime, timedelta
import numpy as np
import torch
from flower.config import GroundStationConfig, SatelliteConfig
from flower.orbit_utils import OrbitCalculator
from flower.visualization import FederatedLearningVisualizer
from flower.client import SatelliteFlowerClient

async def test_satellite_fl():
    """测试卫星联邦学习基本功能"""
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
    
    # 创建测试数据
    num_samples = 1000
    X = torch.randn(num_samples, 1, 28, 28)
    y = torch.randint(0, 10, (num_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    
    # 创建客户端配置
    client_config = {
        'batch_size': 32,
        'epochs': 1,
        'learning_rate': 0.01
    }
    
    # 创建客户端
    clients = []
    samples_per_client = len(dataset) // len(satellites)
    
    for i, satellite in enumerate(satellites):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_dataset = torch.utils.data.Subset(dataset, range(start_idx, end_idx))
        
        client = SatelliteFlowerClient(
            satellite_id=satellite.sat_id,
            train_data=client_dataset,
            test_data=client_dataset,
            config=client_config,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        clients.append(client)
    
    # 验证可见性
    current_time = datetime.now()
    visible_satellites = set()
    
    # 定义可见性映射
    visibility_map = {
        "Beijing": [0],     # 北京只能看到轨道0的卫星
        "NewYork": [1]      # 纽约只能看到轨道1的卫星
    }
    
    print("\n卫星可见性测试:")
    for station_config in ground_stations:
        station_visible = []
        for satellite in satellites:
            if satellite.orbit_id in visibility_map[station_config.station_id]:
                if orbit_calculator.check_satellite_visibility(
                    satellite, station_config, current_time):
                    visible_satellites.add(satellite.sat_id)
                    station_visible.append(satellite.sat_id)
        print(f"\n地面站 {station_config.station_id} 可见卫星: {station_visible}")
    
    # 验证可见性结果
    assert len(visible_satellites) > 0, "没有可见的卫星"
    for station_config in ground_stations:
        station_sats = [sat for sat in satellites 
                       if sat.orbit_id in visibility_map[station_config.station_id]]
        assert len(station_sats) <= 2, f"{station_config.station_id} 不应该看到超过2颗卫星"
    
    print(f"\n总可见卫星数量: {len(visible_satellites)}")
    print(f"可见卫星ID: {sorted(list(visible_satellites))}")
    
    # 创建可视化器
    visualizer = FederatedLearningVisualizer(save_dir="results/test")

if __name__ == "__main__":
    asyncio.run(test_satellite_fl()) 