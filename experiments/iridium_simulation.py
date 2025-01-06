import asyncio
import torch
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator, AdvancedOrbitCalculator
from flower.ground_station import GroundStation
from flower.fl_server import SatelliteStrategy
from flower.client import SatelliteFlowerClient, Net
from flower.visualization import FederatedLearningVisualizer
from torchvision import datasets, transforms
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.common import Context  # 添加这行导入
import os

async def create_iridium_constellation(orbit_calculator=None):
    """创建铱星星座配置"""
    satellites = []
    if orbit_calculator is None:
        orbit_calculator = OrbitCalculator(debug_mode=True)
    
    earth_radius = orbit_calculator.earth_radius
    
    # 铱星星座参数
    num_planes = 6        # 6个轨道面
    sats_per_plane = 11   # 每个轨道面11颗卫星
    altitude = 780.0      # 轨道高度780km
    inclination = 86.4    # 轨道倾角86.4度
    
    for plane_id in range(num_planes):
        # 每个轨道面的升交点赤经间隔60度
        raan = (plane_id * 360.0 / num_planes)  
        for sat_id in range(sats_per_plane):
            # 同一轨道面内卫星间隔
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
    
    print(f"创建了 {len(satellites)} 颗卫星")
    return satellites  # 只返回卫星列表

async def create_ground_stations(orbit_calculator):
    """创建地面站"""
    ground_stations = []
    # 添加海拔高度信息 (meters)
    locations = [
        ("Beijing", 39.9042, 116.4074, 44),    # 北京平均海拔
        ("NewYork", 40.7128, -74.0060, 10),    # 纽约平均海拔
        ("London", 51.5074, -0.1278, 11),      # 伦敦平均海拔
        ("Sydney", -33.8688, 151.2093, 39),    # 悉尼平均海拔
        ("Moscow", 55.7558, 37.6173, 156),     # 莫斯科平均海拔
        ("SaoPaulo", -23.5505, -46.6333, 760)  # 圣保罗平均海拔
    ]
    
    for name, lat, lon, alt in locations:
        config = GroundStationConfig(
            station_id=name,
            latitude=lat,
            longitude=lon,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=5,
            altitude=alt/1000.0  # 转换为千米
        )
        ground_stations.append(config)
    
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
    
    for i, sat_config in enumerate(satellites):
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
            satellite_id=sat_config.sat_id,
            train_data=train_subset,
            test_data=test_subset,
            device=device,
            config=sat_config  # 添加卫星配置
        )
        clients.append(client)
    
    return clients

async def run_simulation(use_grouping=False, use_advanced_orbit=False):
    """运行联邦学习仿真"""
    # 禁用Ray的日志重复
    os.environ["RAY_DEDUP_LOGS"] = "0"
    
    print("开始联邦学习测试...")
    
    # 创建轨道计算器
    if use_advanced_orbit:
        orbit_calculator = AdvancedOrbitCalculator(debug_mode=True)
    else:
        orbit_calculator = OrbitCalculator(debug_mode=True)
    
    # 创建卫星配置
    satellites = await create_iridium_constellation(orbit_calculator)
    
    # 创建地面站
    ground_stations = await create_ground_stations(orbit_calculator)
    
    # 创建可视化器
    visualizer = FederatedLearningVisualizer(save_dir="results")
    
    # 创建客户端管理器
    client_manager = SimpleClientManager()
    
    # 创建策略
    strategy = SatelliteStrategy(
        orbit_calculator=orbit_calculator,
        ground_stations=ground_stations,
        fraction_fit=0.2,
        fraction_evaluate=0.15,
        min_fit_clients=8,
        min_evaluate_clients=6,
        min_available_clients=15,
        visualizer=visualizer,
        debug_mode=True  # 启用调试输出
    )
    
    # 创建服务器
    server = fl.server.Server(
        client_manager=client_manager,
        strategy=strategy
    )
    
    # 加载数据集
    train_dataset, test_dataset = await load_dataset()
    
    # 创建客户端生成函数
    def client_fn(context: Context) -> fl.client.Client:
        """创建客户端实例"""
        # 使用 node_id 的哈希值来生成一个确定的索引
        node_id_str = str(context.node_id)
        hash_value = hash(node_id_str)
        idx = abs(hash_value) % len(satellites)  # 使用模运算确保索引在有效范围内
        
        print(f"Creating client for node_id: {context.node_id}, mapped to index: {idx}")
        
        # 计算数据分片
        samples_per_client = len(train_dataset) // len(satellites)
        test_samples_per_client = len(test_dataset) // len(satellites)
        
        # 训练数据子集
        train_start_idx = idx * samples_per_client
        train_end_idx = train_start_idx + samples_per_client
        train_subset = torch.utils.data.Subset(
            train_dataset, 
            range(train_start_idx, train_end_idx)
        )
        
        # 测试数据子集
        test_start_idx = idx * test_samples_per_client
        test_end_idx = test_start_idx + test_samples_per_client
        test_subset = torch.utils.data.Subset(
            test_dataset,
            range(test_start_idx, test_end_idx)
        )
        
        # 创建客户端
        client = SatelliteFlowerClient(
            satellite_id=satellites[idx].sat_id,
            train_data=train_subset,
            test_data=test_subset,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            config=satellites[idx]
        )
        
        # 转换为标准客户端
        return client.to_client()
    
    # 启动模拟
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(satellites),
        client_resources={"num_cpus": 2},
        server=server,
        config=fl.server.ServerConfig(
            num_rounds=15,
            round_timeout=None
        )
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-grouping', action='store_true', 
                       help='是否使用分组训练模式')
    parser.add_argument('--use-advanced-orbit', action='store_true', 
                       help='是否使用高级轨道计算器')
    args = parser.parse_args()
    
    asyncio.run(run_simulation(use_grouping=args.use_grouping, use_advanced_orbit=args.use_advanced_orbit)) 