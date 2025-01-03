import asyncio
import torch
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
from flower.ground_station import GroundStation
from flower.fl_server import SatelliteFlowerServer, SatelliteFedAvg
from flower.client import SatelliteFlowerClient, Net
from flower.group_strategy import GroupedSatelliteFedAvg
from flower.visualization import FederatedLearningVisualizer
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
    return satellites  # 只返回卫星列表

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
            train_data=train_subset,
            test_data=test_subset,
            device=device
        )
        clients.append(client)
    
    return clients

async def run_simulation(use_grouping: bool = False):
    """运行铱星星座联邦学习仿真"""
    try:
        # 创建铱星星座
        orbit_calculator = OrbitCalculator(debug_mode=True)
        satellites = await create_iridium_constellation(orbit_calculator)  # 不再解包返回值
        
        # 创建地面站配置
        station_configs = [
            GroundStationConfig("Beijing", 39.9042, 116.4074, 2000, 10.0),
            GroundStationConfig("NewYork", 40.7128, -74.0060, 2000, 10.0),
            GroundStationConfig("London", 51.5074, -0.1278, 2000, 10.0),
            GroundStationConfig("Sydney", -33.8688, 151.2093, 2000, 10.0),
            GroundStationConfig("Moscow", 55.7558, 37.6173, 2000, 10.0),
            GroundStationConfig("SaoPaulo", -23.5505, -46.6333, 2000, 10.0)
        ]
        
        # 创建可视化器
        visualizer = FederatedLearningVisualizer(save_dir="results")
        
        # 创建策略
        if use_grouping:
            strategy = GroupedSatelliteFedAvg(
                visualizer=visualizer,
                group_size=3,
                use_grouping=True
            )
        else:
            strategy = SatelliteFedAvg(visualizer=visualizer)
        
        # 创建服务器
        server = SatelliteFlowerServer(
            strategy=strategy,
            orbit_calculator=orbit_calculator,
            ground_stations=station_configs,  # 传入配置列表
            num_rounds=5,
            min_fit_clients=3,
            min_eval_clients=3,
            min_available_clients=3,
            visualizer=visualizer
        )
        
        # 创建客户端
        clients = []
        for sat in satellites:
            # 创建训练和测试数据集
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
            
            # 为每个卫星分配数据子集
            samples_per_client = len(train_dataset) // len(satellites)
            test_samples_per_client = len(test_dataset) // len(satellites)
            
            train_start_idx = sat.sat_id * samples_per_client
            train_end_idx = train_start_idx + samples_per_client
            train_subset = torch.utils.data.Subset(
                train_dataset, 
                range(train_start_idx, train_end_idx)
            )
            
            test_start_idx = sat.sat_id * test_samples_per_client
            test_end_idx = test_start_idx + test_samples_per_client
            test_subset = torch.utils.data.Subset(
                test_dataset,
                range(test_start_idx, test_end_idx)
            )
            
            # 创建数据加载器
            train_loader = torch.utils.data.DataLoader(
                train_subset,
                batch_size=32,
                shuffle=True
            )
            
            test_loader = torch.utils.data.DataLoader(
                test_subset,
                batch_size=32,
                shuffle=False
            )
            
            # 创建客户端
            client = SatelliteFlowerClient(
                satellite_id=sat.sat_id,
                train_data=train_loader,
                test_data=test_loader,
                device=torch.device("cpu"),
                config=sat
            )
            clients.append(client)
        
        # 开始训练
        metrics_history = []
        for round in range(5):  # 5轮训练
            try:
                metrics = await server.train_round(clients)
                if metrics:
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
        raise e  # 添加这行以显示完整的错误堆栈

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-grouping', action='store_true', 
                       help='是否使用分组训练模式')
    args = parser.parse_args()
    
    asyncio.run(run_simulation(use_grouping=args.use_grouping)) 