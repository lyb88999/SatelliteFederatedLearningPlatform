# client.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import flwr as fl
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import asyncio
import websockets
import json
from datetime import datetime
import time
import psutil
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
import os
from .resource_monitor import ResourceMonitor
import random

# 在文件开头添加自定义异常类
class SatelliteError(Exception):
    """卫星相关错误的基类"""
    pass

class CommunicationError(SatelliteError):
    """通信错误"""
    pass

class ResourceError(SatelliteError):
    """资源不足错误"""
    pass

class CommunicationWindowExceeded(SatelliteError):
    """通信窗口超时错误"""
    pass

# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, device, epochs=1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        print(f"Training epoch {epoch+1}/{epochs}")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                curr_acc = 100. * correct / total
                print(f"Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}, Accuracy: {curr_acc:.2f}%")
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        total_loss += avg_loss
    
    return total_loss / epochs, correct / total

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

class StatusReporter:
    def __init__(self, client_id: str, websocket_url: str = "ws://localhost:8765"):
        self.client_id = client_id
        self.websocket_url = websocket_url
        self.parameters_version = 0
        
    async def send_status(self, round_number: int, loss: float, accuracy: float, samples: int):
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                status = {
                    "client_id": str(self.client_id),
                    "round": round_number,
                    "loss": float(loss),
                    "accuracy": float(accuracy),
                    "parameters_version": self.parameters_version,
                    "training_samples": int(samples),
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send(json.dumps(status))
                print(f"Status sent: {status}")
        except Exception as e:
            print(f"Error sending status: {e}")
        finally:
            self.parameters_version += 1

# 首先定义 OrbitCoordinatorClient 类
class OrbitCoordinatorClient(fl.client.NumPyClient):
    def __init__(self, cid, model, device, config: SatelliteConfig):
        super().__init__()
        self.cid = cid
        self.model = model
        self.device = device
        self.config = config
        self.orbit_calculator = OrbitCalculator(config)
        self.collected_parameters = []  # 存储收到的参数
        self.collected_metrics = []     # 存储收到的指标
        self.orbit_members = set()      # 轨道内的成员节点
        self.criterion = nn.CrossEntropyLoss()  # 添加损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # 添加优化器
        
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """获取模型参数"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def collect_parameters(self, parameters, num_samples: int, metrics: Dict):
        """收集轨道内其他节点的参数"""
        self.collected_parameters.append((parameters, num_samples))
        self.collected_metrics.append(metrics)
        
    def aggregate_parameters(self, parameters: List[List[np.ndarray]]) -> List[np.ndarray]:
        """聚合收集到的参数"""
        if not parameters:
            return None
            
        # 使用 FedAvg 聚合参数
        num_layers = len(parameters[0])
        aggregated = []
        
        for layer_idx in range(num_layers):
            layer_updates = [p[layer_idx] for p in parameters]
            aggregated.append(np.mean(layer_updates, axis=0))
        
        return aggregated
        
    def fit(self, parameters, config):
        """协调者不进行训练,只进行参数聚合"""
        try:
            print(f"\n{'='*50}")
            print(f"Coordinator {self.cid}: 开始聚合轮次")
            print(f"{'='*50}")
            
            # 设置当前参数
            self.set_parameters(parameters)
            
            # 如果没有收集到参数,返回当前参数
            if not self.collected_parameters:
                print(f"Coordinator {self.cid}: 没有收到任何参数,返回当前参数")
                return parameters, 1, {}
            
            # 聚合参数
            collected_params = [p for p, _ in self.collected_parameters]
            aggregated_parameters = self.aggregate_parameters(collected_params)
            if aggregated_parameters is None:
                print(f"Coordinator {self.cid}: 聚合失败,返回当前参数")
                return parameters, 1, {}
            
            total_samples = sum(num for _, num in self.collected_parameters)
            print(f"Coordinator {self.cid}: 聚合了 {len(self.collected_parameters)} 个客户端的参数")
            print(f"Coordinator {self.cid}: 总样本数: {total_samples}")
            
            # 清空收集的参数
            self.collected_parameters = []
            self.collected_metrics = []
            
            return aggregated_parameters, total_samples, {
                "aggregated_metrics": self.collected_metrics
            }
            
        except Exception as e:
            print(f"Coordinator {self.cid}: 聚合错误 - {str(e)}")
            return parameters, 1, {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """评估模型"""
        self.set_parameters(parameters)
        self.model.eval()
        
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        return loss, total, {"accuracy": accuracy}

    def train_local(self) -> Dict:
        """协调者的本地训练"""
        print(f"Coordinator {self.cid}: 跳过本地训练")
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "parameters": self.get_parameters({})
        }

def create_model() -> nn.Module:
    """创建一个简单的CNN模型"""
    return Net()  # 使用之前定义的 Net 类

# 然后定义 SatelliteFlowerClient 类
class SatelliteFlowerClient(fl.client.Client):
    """卫星端 Flower 客户端"""
    
    def __init__(self, cid: str, model: OrderedDict, config: SatelliteConfig, 
                 train_loader=None, test_loader=None, device=None):
        self.cid = cid
        self.model = model
        self.config = config
        self.optimizer = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.training_progress = {}
        
        # 如果是协调者，不需要训练和测试数据
        if self.config.is_coordinator:
            self.train_loader = None
            self.test_loader = None
            
    def get_parameters(self, config) -> List[np.ndarray]:
        """获取模型参数"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """设置模型参数"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def receive_model(self, parameters) -> None:
        """接收模型参数（协调者专用）"""
        if not self.config.is_coordinator:
            raise ValueError(f"客户端 {self.cid} 不是协调者，无法接收模型")
            
        # 将参数转换为 numpy 数组列表
        parameters_arrays = fl.common.parameters_to_ndarrays(parameters)
        # 设置模型参数
        self.set_parameters(parameters_arrays)
        
    def fit(self, parameters, config) -> Tuple[fl.common.Parameters, int, Dict]:
        """训练模型"""
        # 设置模型参数
        self.set_parameters(fl.common.parameters_to_ndarrays(parameters))
        
        # 如果是协调者，只接收参数不训练
        if self.config.is_coordinator or config.get("is_coordinator", False):
            return parameters, 0, {}  # 直接返回接收到的参数
        
        if self.train_loader is None:
            print(f"警告: 客户端 {self.cid} 没有训练数据")
            return parameters, 0, {}
        
        # 初始化优化器
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        
        # 训练模型
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        accuracy = correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        print(f"客户端 {self.cid} 训练完成: accuracy={accuracy:.4f}, loss={avg_loss:.4f}")
        
        # 返回 Parameters 对象而不是列表
        parameters = fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        
        return parameters, total, {
            "accuracy": float(accuracy),
            "loss": float(avg_loss)
        }

    def evaluate(self, parameters, config):
        """评估模型"""
        try:
            # 先将 Parameters 对象转换为 numpy 数组列表
            parameters_arrays = fl.common.parameters_to_ndarrays(parameters)
            self.set_parameters(parameters_arrays)
            
            if self.test_loader is None:
                print(f"警告: 客户端 {self.cid} 没有测试数据")
                return None
            
            # 设置评估模式
            self.model.eval()
            total_loss = 0.0
            total_examples = 0
            correct = 0
            
            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                    
                    total_loss += loss.item() * len(data)
                    total_examples += len(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            # 计算指标
            accuracy = correct / total_examples
            avg_loss = total_loss / total_examples
            
            metrics = {
                "accuracy": float(accuracy),
                "loss": float(avg_loss)
            }
            
            print(f"客户端 {self.cid} 评估完成: accuracy={accuracy:.4f}, loss={avg_loss:.4f}")
            
            return float(total_loss), total_examples, metrics
            
        except Exception as e:
            print(f"客户端 {self.cid} 评估失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

class AsyncSatelliteClient(fl.client.NumPyClient):
    async def _train_async(self):
        """异步训练实现"""
        try:
            self._check_resources()
            if not self._check_communication_window():
                raise CommunicationError("不在通信窗口内")
                
            # 训练过程
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            loss, accuracy = train(self.model, self.train_loader, optimizer, self.device, epochs=1)
            
            return self.get_parameters({}), len(self.train_loader.dataset), {
                "loss": float(loss),
                "accuracy": float(accuracy)
            }
            
        except Exception as e:
            self._save_checkpoint()
            raise e
            
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_progress': self.training_progress
        }
        torch.save(checkpoint, f'checkpoint_satellite_{self.cid}.pt')

def start_client(cid: int, orbit_id: int, is_coordinator: bool = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
    testset = datasets.MNIST('data', train=False, transform=transform)
    
    # 使用更好的数据分配方式
    n_samples = len(trainset)
    n_clients = 3  # 假设最多有3个客户端
    samples_per_client = n_samples // n_clients
    
    # 计算当前客户端的数据范围
    start_idx = cid * samples_per_client
    end_idx = start_idx + samples_per_client if cid < n_clients - 1 else n_samples
    indices = list(range(start_idx, end_idx))
    
    # 随机打乱索引
    np.random.shuffle(indices)
    
    print(f"Client {cid}:")
    print(f"- Total dataset size: {n_samples}")
    print(f"- Assigned samples: {len(indices)}")
    
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=testset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    if len(train_loader) == 0:
        raise ValueError(f"No training data for client {cid}")

    # 创建卫星配置
    ground_stations = [
        GroundStationConfig(
            "Beijing", 
            39.9042, 
            116.4074, 
            coverage_radius=2000,
            min_elevation=10.0
        ),
        GroundStationConfig(
            "Shanghai", 
            31.2304, 
            121.4737, 
            coverage_radius=2000,
            min_elevation=10.0
        ),
        GroundStationConfig(
            "Guangzhou", 
            23.1291, 
            113.2644, 
            coverage_radius=2000,
            min_elevation=10.0
        )
    ]
    
    satellite_config = SatelliteConfig(
        orbit_altitude=550.0,
        orbit_inclination=97.6,
        orbital_period=95,
        ground_stations=ground_stations,
        ascending_node=0.0,
        mean_anomaly=0.0,
        orbit_id=orbit_id,
        is_coordinator=is_coordinator
    )

    model = Net().to(device)
    if is_coordinator:
        client = OrbitCoordinatorClient(
            cid=cid,
            model=model,
            device=device,
            config=satellite_config
        )
    else:
        client = SatelliteFlowerClient(
            cid=cid,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            config=satellite_config
        )
    
    # 设置为调试模式
    client.orbit_calculator.debug_mode = True
    
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        grpc_max_message_length=1024*1024*1024
    )

if __name__ == "__main__":
    import sys
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    orbit_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    is_coordinator = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    start_client(client_id, orbit_id, is_coordinator)