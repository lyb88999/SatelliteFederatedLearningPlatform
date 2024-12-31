# client.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import flwr as fl
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple
import asyncio
import websockets
import json
from datetime import datetime
import time
import psutil
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
import os

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
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
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
        
    def collect_parameters(self, parameters, num_samples: int, metrics: Dict):
        """收集轨道内其他节点的参数"""
        self.collected_parameters.append((parameters, num_samples))
        self.collected_metrics.append(metrics)
        
    def aggregate_parameters(self):
        """聚合收集到的参数"""
        if not self.collected_parameters:
            return None
            
        # 使用 FedAvg 聚合参数
        total_samples = sum(num for _, num in self.collected_parameters)
        weighted_params = [
            [layer * num for layer in params] 
            for params, num in self.collected_parameters
        ]
        
        aggregated = [
            sum(layer_updates) / total_samples 
            for layer_updates in zip(*weighted_params)
        ]
        
        # 清空收集的参数
        self.collected_parameters = []
        self.collected_metrics = []
        
        return aggregated
        
    def fit(self, parameters, config):
        """协调者不进行训练，只进行参数聚合"""
        try:
            print(f"\n{'='*50}")
            print(f"Coordinator {self.cid}: 开始聚合轮次")
            print(f"{'='*50}")
            
            # 如果没有收集到参数，返回原始参数
            if not self.collected_parameters:
                print(f"Coordinator {self.cid}: 没有收到任何参数，返回原始参数")
                return parameters, 1, {}  # 返回 1 作为样本数，避免除零错误
            
            # 聚合参数
            aggregated_parameters = self.aggregate_parameters()
            if aggregated_parameters is None:
                print(f"Coordinator {self.cid}: 聚合失败，返回原始参数")
                return parameters, 1, {}
            
            total_samples = sum(num for _, num in self.collected_parameters)
            print(f"Coordinator {self.cid}: 聚合了 {len(self.collected_parameters)} 个客户端的参数")
            print(f"Coordinator {self.cid}: 总样本数: {total_samples}")
            
            return aggregated_parameters, total_samples, {
                "aggregated_metrics": self.collected_metrics
            }
            
        except Exception as e:
            print(f"Coordinator {self.cid}: 聚合错误 - {str(e)}")
            return parameters, 1, {}  # 出错时返回原始参数

# 然后定义 SatelliteFlowerClient 类
class SatelliteFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train_loader, test_loader, device, config: SatelliteConfig):
        super().__init__()
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        self.orbit_calculator = OrbitCalculator(config)
        self.current_ground_station = None
        self.next_window = None
        self.training_buffer = []
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.coordinator = None  # 协调者引用
        
    def _wait_for_communication_window(self) -> bool:
        """等待下一个通信窗口"""
        print(f"\n{'='*50}")
        print(f"Client {self.cid}: 通信窗口检查")
        print(f"{'='*50}")
        print(f"当前时间: {datetime.now().strftime('%H:%M:%S')}")
        
        # 检查所有地面站
        for station in self.config.ground_stations:
            is_visible = self.orbit_calculator.calculate_visibility(station, datetime.now())
            print(f"\n检查地面站 {station.station_id}:")
            print(f"- 位置: {station.latitude:.2f}°N, {station.longitude:.2f}°E")
            print(f"- 可见性: {'可见' if is_visible else '不可见'}")
            
            if is_visible:
                self.current_ground_station = station
                print(f"\n✅ 已建立与地面站 {station.station_id} 的连接")
                return True
        
        # 如果没有可用的地面站，预测下一个窗口
        print("\n❌ 当前无可用地面站，计算下一个通信窗口...")
        next_windows = []
        for station in self.config.ground_stations:
            window_time, duration = self.orbit_calculator.get_next_window(
                station,
                datetime.now()
            )
            if window_time:
                next_windows.append((window_time, duration, station))
                print(f"\n地面站 {station.station_id} 的下一个窗口:")
                print(f"- 开始时间: {window_time.strftime('%H:%M:%S')}")
                print(f"- 持续时间: {duration.total_seconds():.1f} 秒")
        
        if next_windows:
            # 选择最近的窗口
            next_window, duration, station = min(next_windows, key=lambda x: x[0])
            wait_time = (next_window - datetime.now()).total_seconds()
            
            if wait_time > 0:
                print(f"\n⏳ 等待下一个通信窗口...")
                print(f"- 选择地面站: {station.station_id}")
                print(f"- 等待时间: {wait_time:.1f} 秒")
                print(f"- 窗口持续: {duration.total_seconds():.1f} 秒")
                
                # 显示倒计时
                for remaining in range(int(wait_time), 0, -1):
                    print(f"\r倒计时: {remaining} 秒...", end='', flush=True)
                    time.sleep(1)
                print("\n")
                
                self.current_ground_station = station
                print(f"✅ 已建立与地面站 {station.station_id} 的连接")
                return True
        
        print("\n❌ 无法找到可用的通信窗口")
        print(f"{'='*50}\n")
        return False

    def _save_checkpoint(self):
        """保存检查点"""
        try:
            checkpoint = {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                'client_id': self.cid,
                'current_ground_station': self.current_ground_station.station_id if self.current_ground_station else None,
                'timestamp': datetime.now().isoformat()
            }
            
            checkpoint_path = f'checkpoint_satellite_{self.cid}.pt'
            torch.save(checkpoint, checkpoint_path)
            print(f"Client {self.cid}: 保存检查点到 {checkpoint_path}")
            
        except Exception as e:
            print(f"Client {self.cid}: 保存检查点失败 - {str(e)}")

    def _load_checkpoint(self):
        """加载检查点"""
        try:
            checkpoint_path = f'checkpoint_satellite_{self.cid}.pt'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                self.model.load_state_dict(checkpoint['model_state'])
                if checkpoint['optimizer_state'] and self.optimizer:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                print(f"Client {self.cid}: 加载检查点 {checkpoint_path}")
                return True
        except Exception as e:
            print(f"Client {self.cid}: 加载检查点失败 - {str(e)}")
        return False

    def set_coordinator(self, coordinator: OrbitCoordinatorClient):
        """设置轨道协调者"""
        self.coordinator = coordinator
        
    def _check_orbit_visibility(self, other_satellite) -> bool:
        """检查是否可以与轨道内其他卫星通信"""
        # TODO: 实现轨道内可见性检查
        return True
        
    def fit(self, parameters, config):
        """训练后将参数发送给协调者"""
        try:
            print(f"\n{'='*50}")
            print(f"Client {self.cid}: 开始训练回合")
            print(f"{'='*50}")
            
            # 检查通信窗口
            if not self._wait_for_communication_window():
                print(f"❌ Client {self.cid}: 无法建立通信连接")
                return parameters, 1, {}  # 返回原始参数而不是抛出异常
            
            # 设置模型参数
            self.set_parameters(parameters)
            
            # 训练模型
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            loss, accuracy = train(self.model, self.train_loader, optimizer, self.device, epochs=1)
            
            print(f"Client {self.cid}: 训练完成 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # 获取更新后的参数
            updated_parameters = self.get_parameters({})
            
            # 如果是轨道内节点且有协调者，发送参数给协调者
            if self.coordinator and self._check_orbit_visibility(self.coordinator):
                self.coordinator.collect_parameters(
                    updated_parameters,
                    len(self.train_loader.dataset),
                    {
                        "loss": float(loss),
                        "accuracy": float(accuracy)
                    }
                )
                print(f"Client {self.cid}: 参数已发送给协调者 {self.coordinator.cid}")
                return parameters, 1, {}  # 返回原始参数，避免服务器聚合
                
            return updated_parameters, len(self.train_loader.dataset), {
                "loss": float(loss),
                "accuracy": float(accuracy)
            }
            
        except Exception as e:
            print(f"Client {self.cid}: 训练错误 - {str(e)}")
            self._save_checkpoint()
            return parameters, 1, {}  # 出错时返回原始参数

    def evaluate(self, parameters, config):
        """评估模型"""
        try:
            print(f"Client {self.cid}: 开始评估")
            
            # 检查通信窗口
            if not self._wait_for_communication_window():
                print(f"Client {self.cid}: 无法建立通信连接")
                raise CommunicationError("无可用通信窗口")
            
            # 设置模型参数
            self.set_parameters(parameters)
            
            # 评估模型
            loss, accuracy = test(self.model, self.test_loader, self.device)
            
            print(f"Client {self.cid}: 评估完成 - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
            # 检查是否仍在通信窗口内
            if not self.orbit_calculator.calculate_visibility(self.current_ground_station, datetime.now()):
                print(f"Client {self.cid}: 通信窗口已关闭，等待下一个窗口")
                if not self._wait_for_communication_window():
                    raise CommunicationError("无法发送评估结果")
            
            return float(loss), len(self.test_loader.dataset), {
                "accuracy": float(accuracy)
            }
            
        except Exception as e:
            print(f"Client {self.cid}: 评估错误 - {str(e)}")
            raise
    
    def set_parameters(self, parameters):
        """设置模型参数"""
        try:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Client {self.cid}: Error setting parameters - {str(e)}")
            raise
    
    def get_parameters(self, config):
        """获取模型参数"""
        try:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        except Exception as e:
            print(f"Client {self.cid}: Error getting parameters - {str(e)}")
            raise

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