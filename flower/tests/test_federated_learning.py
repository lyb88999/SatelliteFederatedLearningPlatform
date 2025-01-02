import pytest
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
from flower.client import SatelliteFlowerClient, OrbitCoordinatorClient, Net
from flower.fl_server import SatelliteFlowerServer
from flower.scheduler import TrainingScheduler
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters
from collections import OrderedDict

class SimpleTestModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def create_test_data():
    """创建测试数据"""
    # 创建随机数据
    train_data = torch.randn(100, 10)
    train_labels = torch.randint(0, 2, (100,))
    test_data = torch.randn(20, 10)
    test_labels = torch.randint(0, 2, (20,))
    
    # 创建数据加载器
    train_loader = DataLoader(
        list(zip(train_data, train_labels)),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        list(zip(test_data, test_labels)),
        batch_size=32,
        shuffle=False
    )
    
    return train_loader, test_loader

@pytest.fixture
def fl_test_environment():
    """创建联邦学习测试环境"""
    # 创建卫星配置
    satellites = []
    ground_stations = [
        GroundStationConfig("Beijing", 39.9042, 116.4074, 2000, 10.0),
        GroundStationConfig("Shanghai", 31.2304, 121.4737, 2000, 10.0)
    ]
    
    # 创建3个轨道，每个轨道4颗卫星
    for orbit_id in range(3):
        for i in range(4):
            sat_config = SatelliteConfig(
                orbit_altitude=550.0 + orbit_id * 50,
                orbit_inclination=97.6,
                orbital_period=95 + orbit_id * 2,
                ground_stations=ground_stations,
                ascending_node=orbit_id * 120.0,
                mean_anomaly=float(i * 90),
                orbit_id=orbit_id,
                sat_id=i,
                is_coordinator=(i == 0)
            )
            satellites.append(sat_config)
    
    # 创建训练数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data', train=False, transform=transform)
    
    # 为每个客户端分配数据
    n_clients = len(satellites) - 3  # 减去协调者数量
    samples_per_client = len(trainset) // n_clients
    
    # 创建模型并初始化参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initial_model = Net().to(device)
    
    # 创建客户端和协调者
    clients = []
    coordinators = []
    
    # 按轨道组织卫星
    satellites_by_orbit = {}
    for sat in satellites:
        if sat.orbit_id not in satellites_by_orbit:
            satellites_by_orbit[sat.orbit_id] = []
        satellites_by_orbit[sat.orbit_id].append(sat)
    
    # 为每个轨道创建客户端
    client_idx = 0
    for orbit_id, orbit_sats in satellites_by_orbit.items():
        for i, sat_config in enumerate(orbit_sats):
            # 为每个客户端创建数据加载器
            if not sat_config.is_coordinator:
                start_idx = client_idx * samples_per_client
                end_idx = start_idx + samples_per_client
                indices = list(range(start_idx, end_idx))
                
                train_loader = DataLoader(
                    dataset=trainset,
                    batch_size=32,
                    sampler=torch.utils.data.SubsetRandomSampler(indices)
                )
                
                test_loader = DataLoader(
                    dataset=testset,
                    batch_size=32,
                    shuffle=False
                )
                
                client = SatelliteFlowerClient(
                    cid=f"orbit_{orbit_id}_sat_{i}",
                    model=initial_model.state_dict(),
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    config=sat_config
                )
                clients.append(client)
                client_idx += 1
            else:
                # 创建协调者
                coordinator = SatelliteFlowerClient(
                    cid=f"orbit_{orbit_id}_sat_{i}",
                    model=initial_model.state_dict(),
                    config=sat_config
                )
                coordinators.append(coordinator)
    
    # 创建轨道计算器
    orbit_calculator = OrbitCalculator(satellite_config=satellites[0])
    
    # 创建训练调度器
    scheduler = TrainingScheduler(orbit_calculator=orbit_calculator)
    
    return {
        'satellites': satellites,
        'clients': clients,
        'coordinators': coordinators,
        'orbit_calculator': orbit_calculator,
        'scheduler': scheduler,
        'initial_model': initial_model
    }

def test_training_schedule(fl_test_environment):
    """测试训练调度"""
    env = fl_test_environment
    start_time = datetime.now()
    
    # 获取协调者ID
    coordinators = {
        sat.orbit_id: f"orbit_{sat.orbit_id}_sat_0"
        for sat in env['satellites']
        if sat.is_coordinator
    }
    
    # 生成资源状态
    resource_states = {}
    for client in env['clients']:
        resource_states[client.cid] = {
            'battery': 80.0,
            'memory': 30.0,
            'cpu': 20.0
        }
    for coordinator in env['coordinators']:
        resource_states[coordinator.cid] = {
            'battery': 90.0,
            'memory': 20.0,
            'cpu': 15.0
        }
    
    # 创建训练调度计划
    schedule = env['scheduler'].create_training_schedule(
        satellites=env['satellites'],
        coordinators=coordinators,
        resource_states=resource_states,
        start_time=start_time,
        duration_hours=1
    )
    
    print("\n" + "="*50)
    print("训练调度计划:")
    print("="*50)
    
    for client_id, windows in schedule.items():
        print(f"\n客户端 {client_id}:")
        for window in windows:
            print(f"- 开始时间: {window.start_time.strftime('%H:%M:%S')}")
            print(f"  持续时间: {window.duration.total_seconds()/60:.1f}分钟")
            print(f"  协调者: {window.coordinator_id}")
            print(f"  优先级: {window.priority}")
    
    # 验证调度计划
    assert len(schedule) > 0
    for windows in schedule.values():
        for window in windows:
            assert isinstance(window.start_time, datetime)
            assert isinstance(window.duration, timedelta)
            assert window.priority in [1, 2, 3]

class SatelliteClientProxy(ClientProxy):
    def __init__(self, cid: str, client: fl.client.Client):
        super().__init__(cid)
        self.client = client
        
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """获取客户端模型参数"""
        return self.client.get_parameters(config)
        
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """执行客户端训练"""
        return self.client.fit(parameters, config)
        
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """执行客户端评估"""
        return self.client.evaluate(parameters, config)

    def get_properties(self, config: Dict[str, str]) -> Dict[str, str]:
        """获取客户端属性"""
        return {}

    def reconnect(self, seconds: Optional[float]) -> bool:
        """重新连接客户端"""
        return True

def test_hierarchical_training(fl_test_environment):
    """测试分层训练"""
    env = fl_test_environment
    
    # 1. 加载 MNIST 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('../data', train=False, transform=transform)
    
    # 2. 为每个客户端分配不同的数据子集
    num_clients = len(env['clients'])
    samples_per_client = len(mnist_train) // num_clients
    
    for idx, client in enumerate(env['clients']):
        # 分配训练数据
        start_idx = idx * samples_per_client
        end_idx = start_idx + samples_per_client
        train_indices = list(range(start_idx, end_idx))
        
        client.train_loader = DataLoader(
            torch.utils.data.Subset(mnist_train, train_indices),
            batch_size=32,
            shuffle=True
        )
        
        # 分配测试数据
        test_indices = list(range(idx * len(mnist_test) // num_clients, 
                                (idx + 1) * len(mnist_test) // num_clients))
        client.test_loader = DataLoader(
            torch.utils.data.Subset(mnist_test, test_indices),
            batch_size=32,
            shuffle=False
        )
    
    # 1. 确保初始模型正确加载
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    initial_model = Net().to(device)
    initial_parameters = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
    print("\n初始模型参数形状:")
    print([arr.shape for arr in initial_parameters])
    
    # 2. 修改策略配置
    def fit_metrics_aggregation_fn(metrics):
        """聚合训练指标"""
        metrics_dicts = [m[1] for m in metrics]
        aggregated = {
            "accuracy": np.mean([m.get("accuracy", 0.0) for m in metrics_dicts]) if metrics_dicts else 0.0,
            "loss": np.mean([m.get("loss", float("inf")) for m in metrics_dicts]) if metrics_dicts else float("inf")
        }
        print(f"\n聚合训练指标: {aggregated}")
        return aggregated
    
    def evaluate_metrics_aggregation_fn(metrics):
        """聚合评估指标"""
        metrics_dicts = [m[1] for m in metrics]
        aggregated = {
            "accuracy": np.mean([m.get("accuracy", 0.0) for m in metrics_dicts]) if metrics_dicts else 0.0,
            "loss": np.mean([m.get("loss", float("inf")) for m in metrics_dicts]) if metrics_dicts else float("inf")
        }
        print(f"\n聚合评估指标: {aggregated}")
        return aggregated
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )
    
    # 3. 为每个客户端创建新的模型实例和训练数据
    for client in env['clients']:
        # 创建新的模型实例
        client.model = Net().to(device)
        client.model.load_state_dict(
            OrderedDict({
                k: torch.tensor(v) 
                for k, v in zip(initial_model.state_dict().keys(), initial_parameters)
            })
        )
        
        # 创建训练数据
        if not hasattr(client, 'train_loader') or client.train_loader is None:
            # 为每个客户端生成不同的随机种子
            seed = hash(client.cid) % 10000
            torch.manual_seed(seed)
            
            train_data = torch.randn(100, 1, 28, 28).to(device)
            train_labels = torch.randint(0, 10, (100,)).to(device)
            client.train_loader = DataLoader(
                TensorDataset(train_data, train_labels),
                batch_size=32,
                shuffle=True
            )
            
        # 创建测试数据
        if not hasattr(client, 'test_loader') or client.test_loader is None:
            # 使用不同的随机种子
            test_seed = (seed + 1) % 10000  # 确保和训练数据的种子不同
            torch.manual_seed(test_seed)
            
            test_data = torch.randn(20, 1, 28, 28).to(device)
            test_labels = torch.randint(0, 10, (20,)).to(device)
            client.test_loader = DataLoader(
                TensorDataset(test_data, test_labels),
                batch_size=32,
                shuffle=False
            )
    
    # 4. 为每个协调者也创建模型和数据
    for coordinator in env['coordinators']:
        coordinator.model = Net().to(device)
        coordinator.model.load_state_dict(
            OrderedDict({
                k: torch.tensor(v)
                for k, v in zip(initial_model.state_dict().keys(), initial_parameters)
            })
        )
        # 协调者不需要训练和测试数据
        coordinator.train_loader = None
        coordinator.test_loader = None
    
    # 5. 创建客户端管理器和服务器
    client_manager = fl.server.SimpleClientManager()
    server = SatelliteFlowerServer(
        client_manager=client_manager,
        strategy=strategy,
        orbit_calculator=env['orbit_calculator']
    )
    
    # 6. 注册客户端和协调者
    for client in env['clients']:
        client_proxy = SatelliteClientProxy(
            cid=client.cid,
            client=client
        )
        client_manager.register(client_proxy)
    
    for coordinator in env['coordinators']:
        coordinator_proxy = SatelliteClientProxy(
            cid=coordinator.cid,
            client=coordinator
        )
        client_manager.register(coordinator_proxy)
    
    print("\n" + "="*50)
    print("开始分层训练:")
    print("="*50)
    print(f"客户端数量: {len(env['clients'])}")
    print(f"协调者数量: {len(env['coordinators'])}")
    
    # 执行训练
    history = server.fit(num_rounds=3)
    
    print("\n训练结果:")
    for round_idx, (accuracy, loss) in enumerate(zip(history['accuracy'], history['loss'])):
        print(f"\n轮次 {round_idx + 1}:")
        print(f"- 准确率: {accuracy:.4f}")
        print(f"- 损失: {loss:.4f}")
    
    # 验证训练结果
    assert len(history['accuracy']) == 3
    assert len(history['loss']) == 3
    assert all(0 <= acc <= 1 for acc in history['accuracy'])
    assert all(loss >= 0 for loss in history['loss'])
    assert any(acc > 0 for acc in history['accuracy']), "所有准确率都为0"

def create_test_environment():
    """创建测试环境"""
    orbit_calculator = OrbitCalculator()
    
    # 创建三个轨道平面
    orbit_params = [
        {
            'semi_major_axis': 7000.0,  # LEO轨道
            'inclination': 98.0,        # 太阳同步轨道
            'raan': i * 120.0           # 均匀分布的轨道平面
        }
        for i in range(3)
    ]
    
    clients = []
    coordinators = []
    
    for orbit_id, params in enumerate(orbit_params):
        # 在每个轨道上均匀分布4颗卫星
        for sat_id in range(4):
            config = SatelliteConfig(
                orbit_id=orbit_id,
                sat_id=sat_id,
                is_coordinator=(sat_id == 0),
                semi_major_axis=params['semi_major_axis'],
                inclination=params['inclination'],
                raan=params['raan'],
                arg_perigee=sat_id * 90.0  # 均匀分布在轨道上
            )
            
            client = SatelliteFlowerClient(
                cid=f"orbit_{orbit_id}_sat_{sat_id}",
                model=create_model(),
                config=config
            )
            
            if config.is_coordinator:
                coordinators.append(client)
            else:
                clients.append(client)
                
    return {
        'clients': clients,
        'coordinators': coordinators,
        'orbit_calculator': orbit_calculator
    }

def create_model():
    """创建一个简单的测试模型"""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    return model

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 