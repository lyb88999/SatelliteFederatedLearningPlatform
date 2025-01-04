import pytest
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from flower.config import SatelliteConfig
from flower.orbit_utils import OrbitCalculator

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

def test_fl_process():
    """测试联邦学习流程"""
    # 1. 创建测试环境
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
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
    
    print("\n联邦学习流程测试:")
    print(f"卫星数量: {len(satellites)}")
    print(f"轨道数量: {len(set(sat.orbit_id for sat in satellites))}")
    
    # 5. 验证卫星配置
    for sat in satellites:
        # 验证轨道高度
        height = sat.semi_major_axis - earth_radius
        assert abs(height - 550.0) < 1.0, f"卫星 {sat.sat_id} 高度不正确"
        
        # 验证轨道倾角
        assert abs(sat.inclination - 97.6) < 0.1, f"卫星 {sat.sat_id} 倾角不正确"
    
    print("\n卫星配置验证通过")
    
    # 6. 验证模型结构
    test_input = torch.randn(1, 1, 28, 28).to(device)
    test_output = initial_model(test_input)
    assert test_output.shape == (1, 10), "模型输出维度不正确"
    
    print("模型结构验证通过")
    
    # 7. 验证数据集
    assert len(mnist_train) > 0, "训练集为空"
    assert len(mnist_test) > 0, "测试集为空"
    
    print(f"数据集验证通过:")
    print(f"- 训练集大小: {len(mnist_train)}")
    print(f"- 测试集大小: {len(mnist_test)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 