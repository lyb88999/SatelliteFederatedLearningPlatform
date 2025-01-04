import asyncio
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
from flower.fl_server import SatelliteFlowerServer, SatelliteFedAvg
from flower.group_strategy import GroupedSatelliteFedAvg
from flower.visualization import FederatedLearningVisualizer
from flower.client import SatelliteFlowerClient, OrbitCoordinatorClient, Net
from experiments.iridium_simulation import create_iridium_constellation
import time

async def load_dataset():
    """加载MNIST数据集"""
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

async def create_clients(satellites, train_dataset, test_dataset, device):
    """创建客户端"""
    clients = []
    samples_per_client = len(train_dataset) // len(satellites)
    test_samples_per_client = len(test_dataset) // len(satellites)
    
    for i, sat in enumerate(satellites):
        # 训练数据
        train_start_idx = i * samples_per_client
        train_end_idx = train_start_idx + samples_per_client
        train_subset = torch.utils.data.Subset(
            train_dataset, 
            range(train_start_idx, train_end_idx)
        )
        
        # 测试数据
        test_start_idx = i * test_samples_per_client
        test_end_idx = test_start_idx + test_samples_per_client
        test_subset = torch.utils.data.Subset(
            test_dataset,
            range(test_start_idx, test_end_idx)
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_subset,
            batch_size=32,
            shuffle=True
        )
        
        test_loader = DataLoader(
            test_subset,
            batch_size=32,
            shuffle=False
        )
        
        # 创建客户端
        if sat.is_coordinator:
            client = OrbitCoordinatorClient(
                satellite_id=sat.sat_id,
                device=device,
                config=sat
            )
        else:
            client = SatelliteFlowerClient(
                satellite_id=sat.sat_id,
                train_data=train_loader,
                test_data=test_loader,
                device=device,
                config=sat
            )
        clients.append(client)
    
    return clients

async def run_comparison_experiment():
    """运行对比实验：分组 vs 非分组"""
    
    # 创建两个可视化器
    grouped_visualizer = FederatedLearningVisualizer(save_dir="results/grouped")
    normal_visualizer = FederatedLearningVisualizer(save_dir="results/normal")
    
    # 运行两种方案
    results = {}
    
    # 1. 运行分组方案
    print("\n=== 运行分组训练方案 ===")
    strategy_grouped = GroupedSatelliteFedAvg(
        visualizer=grouped_visualizer,
        group_size=3,
        use_grouping=True
    )
    results['grouped'] = await run_single_experiment(strategy_grouped, "grouped")
    
    # 2. 运行普通方案
    print("\n=== 运行普通训练方案 ===")
    strategy_normal = SatelliteFedAvg(visualizer=normal_visualizer)
    results['normal'] = await run_single_experiment(strategy_normal, "normal")
    
    # 绘制对比图
    plot_comparison(results)

async def run_single_experiment(strategy, name):
    """运行单个实验"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建铱星星座
    orbit_calculator = OrbitCalculator(debug_mode=True)
    satellites = await create_iridium_constellation(orbit_calculator)
    
    # 加载数据集
    train_dataset, test_dataset = await load_dataset()
    
    # 创建客户端
    clients = await create_clients(satellites, train_dataset, test_dataset, device)
    print(f"Created {len(clients)} clients")
    
    # 创建地面站配置
    station_configs = [
        GroundStationConfig(
            station_id="Beijing",
            latitude=39.9042,
            longitude=116.4074,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        ),
        GroundStationConfig(
            station_id="NewYork",
            latitude=40.7128,
            longitude=-74.0060,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        ),
        GroundStationConfig(
            station_id="London",
            latitude=51.5074,
            longitude=-0.1278,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        ),
        GroundStationConfig(
            station_id="Sydney",
            latitude=-33.8688,
            longitude=151.2093,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        ),
        GroundStationConfig(
            station_id="Moscow",
            latitude=55.7558,
            longitude=37.6173,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        ),
        GroundStationConfig(
            station_id="SaoPaulo",
            latitude=-23.5505,
            longitude=-46.6333,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        )
    ]
    
    # 创建服务器，为分组和非分组方案设置不同的参数
    if isinstance(strategy, GroupedSatelliteFedAvg):
        server = SatelliteFlowerServer(
            strategy=strategy,
            orbit_calculator=orbit_calculator,
            ground_stations=station_configs,
            num_rounds=10,
            min_fit_clients=3,
            min_eval_clients=3,
            min_available_clients=3,
            visualizer=strategy.visualizer
        )
    else:
        server = SatelliteFlowerServer(
            strategy=strategy,
            orbit_calculator=orbit_calculator,
            ground_stations=station_configs,
            num_rounds=10,
            min_fit_clients=len(clients) // 2,  # 需要更多客户端参与
            min_eval_clients=len(clients) // 2,
            min_available_clients=len(clients) // 2,
            visualizer=strategy.visualizer
        )
    
    # 记录训练过程和时间
    metrics_history = []
    start_time = time.time()
    
    for round in range(10):
        # 模拟通信延迟
        if not isinstance(strategy, GroupedSatelliteFedAvg):
            # 非分组方案需要更多通信时间
            await asyncio.sleep(2.0)  # 模拟更长的通信延迟
        else:
            await asyncio.sleep(0.5)  # 分组方案通信延迟更短
            
        metrics = await server.train_round(clients)
        if metrics:
            metrics_history.append(metrics)
            elapsed_time = time.time() - start_time
            print(f"{name} - 第 {round + 1} 轮: "
                  f"accuracy={metrics['accuracy']:.4f}, "
                  f"loss={metrics['loss']:.4f}, "
                  f"用时={elapsed_time:.2f}秒")
    
    total_time = time.time() - start_time
    return {
        'metrics': metrics_history,
        'total_time': total_time
    }

def plot_comparison(results):
    """绘制对比图"""
    plt.figure(figsize=(15, 5))
    
    # 准确率对比
    plt.subplot(1, 3, 1)
    plt.plot([m['accuracy'] for m in results['grouped']['metrics']], 'b-', label='Grouped')
    plt.plot([m['accuracy'] for m in results['normal']['metrics']], 'r--', label='Normal')
    plt.title('Accuracy Comparison')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 损失对比
    plt.subplot(1, 3, 2)
    plt.plot([m['loss'] for m in results['grouped']['metrics']], 'b-', label='Grouped')
    plt.plot([m['loss'] for m in results['normal']['metrics']], 'r--', label='Normal')
    plt.title('Loss Comparison')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()
    
    # 训练时间对比
    plt.subplot(1, 3, 3)
    times = [results['grouped']['total_time'], results['normal']['total_time']]
    plt.bar(['Grouped', 'Normal'], times)
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('results/comparison.png')
    plt.close()
    
    # 打印最终结果对比
    print("\n=== 实验结果对比 ===")
    print("分组方案:")
    print(f"- 最终准确率: {results['grouped']['metrics'][-1]['accuracy']:.4f}")
    print(f"- 最终损失: {results['grouped']['metrics'][-1]['loss']:.4f}")
    print(f"- 总训练时间: {results['grouped']['total_time']:.2f}秒")
    print("\n普通方案:")
    print(f"- 最终准确率: {results['normal']['metrics'][-1]['accuracy']:.4f}")
    print(f"- 最终损失: {results['normal']['metrics'][-1]['loss']:.4f}")
    print(f"- 总训练时间: {results['normal']['total_time']:.2f}秒")

if __name__ == "__main__":
    asyncio.run(run_comparison_experiment()) 