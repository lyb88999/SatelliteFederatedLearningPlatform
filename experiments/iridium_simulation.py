import asyncio
import torch
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
from flower.ground_station import GroundStation
from flower.fl_server import SatelliteFlowerServer
from flower.client import SatelliteFlowerClient
from torchvision import datasets, transforms

async def create_iridium_constellation(orbit_calculator=None):
    """创建铱星星座配置"""
    satellites = []
    if orbit_calculator is None:
        orbit_calculator = OrbitCalculator(debug_mode=True)
    
    earth_radius = orbit_calculator.earth_radius
    
    # 铱星星座参数
    num_planes = 6
    sats_per_plane = 11
    altitude = 780.0  # km
    inclination = 86.4  # 度
    
    for plane_id in range(num_planes):
        raan = (plane_id * 360.0 / num_planes)  # 升交点赤经
        for sat_id in range(sats_per_plane):
            arg_perigee = (sat_id * 360.0 / sats_per_plane)  # 近地点幅角
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
    
    print(f"创建了 {len(satellites)} 颗卫星")
    return satellites, orbit_calculator

async def create_ground_stations(orbit_calculator):
    """创建地面站"""
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
            max_range=2000.0,  # 降低最大通信距离
            min_elevation=10.0,  # 提高最小仰角要求
            max_satellites=5     # 限制同时通信的卫星数
        )
        ground_stations.append(GroundStation(config, orbit_calculator))
    
    print(f"创建了 {len(ground_stations)} 个地面站")
    return ground_stations

async def load_dataset():
    """加载并预处理MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        './data', train=False,
        transform=transform
    )
    
    return train_dataset, test_dataset

async def create_clients(satellites, train_dataset, test_dataset):
    """创建联邦学习客户端"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clients = []
    
    # 为每个卫星分配数据
    samples_per_client = len(train_dataset) // len(satellites)
    test_samples_per_client = len(test_dataset) // len(satellites)
    
    for i, sat in enumerate(satellites):
        # 训练数据子集
        train_start_idx = i * samples_per_client
        train_end_idx = train_start_idx + samples_per_client
        train_subset = torch.utils.data.Subset(
            train_dataset, 
            range(train_start_idx, train_end_idx)
        )
        
        # 测试数据子集
        test_start_idx = i * test_samples_per_client
        test_end_idx = test_start_idx + test_samples_per_client
        test_subset = torch.utils.data.Subset(
            test_dataset,
            range(test_start_idx, test_end_idx)
        )
        
        # 创建客户端
        client = SatelliteFlowerClient(
            satellite_id=sat.sat_id,
            train_dataset=train_subset,
            test_dataset=test_subset,
            device=device
        )
        clients.append(client)
    
    return clients

async def run_simulation():
    """运行铱星联邦学习仿真"""
    try:
        # 1. 创建星座
        orbit_calculator = OrbitCalculator(debug_mode=True)  # 设置为调试模式
        satellites, _ = await create_iridium_constellation(orbit_calculator)
        
        # 2. 创建地面站
        ground_stations = await create_ground_stations(orbit_calculator)
        
        # 3. 加载数据集
        train_dataset, test_dataset = await load_dataset()
        
        # 4. 创建客户端
        clients = await create_clients(satellites, train_dataset, test_dataset)
        
        # 5. 创建服务器
        num_rounds = 10  # 增加训练轮数
        server = SatelliteFlowerServer(
            satellites=satellites,
            ground_stations=ground_stations,
            orbit_calculator=orbit_calculator,
            num_rounds=num_rounds
        )
        
        # 6. 运行训练
        metrics_history = []
        
        for round in range(num_rounds):
            try:
                print(f"\n开始第 {round + 1} 轮训练...")
                metrics = await server.train_round(clients)
                
                if metrics['accuracy'] > 0 or metrics['loss'] < float('inf'):
                    metrics_history.append(metrics)
                    print(f"第 {round + 1} 轮完成: "
                          f"accuracy={metrics['accuracy']:.4f}, "
                          f"loss={metrics['loss']:.4f}")
                else:
                    print(f"第 {round + 1} 轮训练无效")
                
                # 保存检查点
                if (round + 1) % 5 == 0:
                    torch.save({
                        'round': round + 1,
                        'model_state': server.model,
                        'metrics': metrics_history
                    }, f'checkpoints/round_{round+1}.pt')
                    
            except Exception as e:
                print(f"轮次 {round + 1} 训练失败: {str(e)}")
                continue
        
        # 保存最终结果
        print("\n保存实验结果...")
        server.visualizer.plot_satellite_metrics()
        server.visualizer.plot_orbit_metrics()
        server.visualizer.plot_global_metrics()
        
        print(f"实验结果已保存到 {server.visualizer.save_dir} 目录")
        
        return metrics_history
        
    except Exception as e:
        print(f"仿真运行失败: {str(e)}")
        return []

if __name__ == "__main__":
    asyncio.run(run_simulation()) 