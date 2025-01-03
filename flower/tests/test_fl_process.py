import pytest
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np
from flower.config import SatelliteConfig
from flower.orbit_utils import OrbitCalculator
from flower.client import SatelliteFlowerClient, Net
from flower.fl_server import SatelliteFlowerServer, SatelliteFedAvg
import flwr as fl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def test_fl_process():
    """测试联邦学习流程"""
    
    # 1. 创建测试环境
    orbit_calculator = OrbitCalculator(debug_mode=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 2. 创建数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('../data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('../data', train=False, transform=transform)
    
    # 3. 创建初始模型
    initial_model = Net().to(device)
    
    # 4. 创建卫星配置
    satellites = []
    for orbit_id in range(3):  # 3个轨道
        for sat_id in range(4):  # 每个轨道4颗卫星
            config = SatelliteConfig(
                orbit_id=orbit_id,
                sat_id=sat_id,
                is_coordinator=(sat_id == 0),  # 每个轨道的第一颗卫星作为协调者
                semi_major_axis=7000.0,
                inclination=98.0,
                raan=orbit_id * 120.0,
                arg_perigee=sat_id * 90.0
            )
            satellites.append(config)
    
    # 5. 创建客户端
    clients = []
    coordinators = []
    samples_per_client = len(mnist_train) // (len(satellites) - 3)  # 减去3个协调者
    
    for sat_config in satellites:
        # 为每个客户端创建一个新的模型实例
        model = Net().to(device)
        model.load_state_dict(initial_model.state_dict())  # 复制初始模型参数
        
        client = SatelliteFlowerClient(
            cid=f"orbit_{sat_config.orbit_id}_sat_{sat_config.sat_id}",
            model=model,  # 添加模型参数
            config=sat_config,
            device=device
        )
        
        if sat_config.is_coordinator:
            coordinators.append(client)
        else:
            # 为工作节点分配数据
            start_idx = len(clients) * samples_per_client
            end_idx = start_idx + samples_per_client
            
            # 训练数据
            train_indices = list(range(start_idx, end_idx))
            client.train_loader = DataLoader(
                torch.utils.data.Subset(mnist_train, train_indices),
                batch_size=64,  # 增加批量大小
                shuffle=True
            )
            
            # 测试数据
            test_indices = list(range(len(clients) * len(mnist_test) // (len(satellites) - 3),
                                    (len(clients) + 1) * len(mnist_test) // (len(satellites) - 3)))
            client.test_loader = DataLoader(
                torch.utils.data.Subset(mnist_test, test_indices),
                batch_size=64,  # 增加批量大小
                shuffle=False
            )
            
            clients.append(client)
    
    # 6. 创建服务器和策略
    strategy = SatelliteFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=len(clients)
    )
    
    server = SatelliteFlowerServer(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy,
        orbit_calculator=orbit_calculator,
        debug_mode=True
    )
    
    # 7. 注册客户端到服务器
    for client in clients + coordinators:
        server.client_manager.register(client)
    
    # 8. 设置初始全局模型
    server.set_global_model(initial_model)
    
    # 9. 模拟训练过程
    print("\n开始模拟联邦学习过程:")
    print(f"- 轨道数量: {len(set(s.orbit_id for s in satellites))}")
    print(f"- 每个轨道的卫星数量: {len(satellites) // 3}")
    print(f"- 协调者数量: {len(coordinators)}")
    print(f"- 工作节点数量: {len(clients)}")
    print(f"- 注册客户端总数: {len(server.client_manager.all())}")
    
    # 开始训练
    for round_idx in range(3):  # 改为3轮
        print(f"\n{'='*20} 轮次 {round_idx+1}/3 {'='*20}")  # 这里也改为3
        
        # 执行训练
        history = server.fit(num_rounds=1)  # 增加训练轮数
        
        # 10. 验证结果
        print("\n训练结果:")
        for round_idx, (accuracy, loss) in enumerate(zip(history['accuracy'], history['loss'])):
            print(f"\n轮次 {round_idx + 1}:")
            print(f"- 准确率: {accuracy:.4f}")
            print(f"- 损失: {loss:.4f}")
            print(f"- 参与训练的客户端数量: {len(server.fit_metrics_aggregated[round_idx])}")
        
        # 验证训练是否成功
        assert len(history['accuracy']) == 1, "应该完成1轮训练"
        assert all(0 <= acc <= 1 for acc in history['accuracy']), "准确率应该在[0,1]范围内"
        assert all(loss >= 0 for loss in history['loss']), "损失值应该非负"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 