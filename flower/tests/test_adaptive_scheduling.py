import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
from flower.client import SatelliteFlowerClient, OrbitCoordinatorClient
import torch
import torch.nn as nn

class SimpleTestModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def create_test_satellites(num_satellites: int = 3) -> List[SatelliteConfig]:
    """创建测试用的卫星配置"""
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
        )
    ]
    
    satellites = []
    for i in range(num_satellites):
        sat_config = SatelliteConfig(
            orbit_altitude=550.0,
            orbit_inclination=97.6,
            orbital_period=95,
            ground_stations=ground_stations,
            ascending_node=0.0,
            mean_anomaly=float(i * 120),  # 均匀分布在轨道上
            orbit_id=1,  # 同一轨道
            is_coordinator=(i == 0)  # 第一颗卫星作为协调者
        )
        satellites.append(sat_config)
    
    return satellites

@pytest.fixture
def test_environment():
    """创建测试环境"""
    satellites = create_test_satellites()
    device = torch.device("cpu")
    
    # 创建客户端
    clients = []
    coordinator = None
    
    for i, sat_config in enumerate(satellites):
        if sat_config.is_coordinator:
            # 创建协调者
            model = SimpleTestModel()
            coordinator = OrbitCoordinatorClient(
                cid=f"sat_{i}",
                model=model,
                device=device,
                config=sat_config
            )
        else:
            # 创建普通客户端
            model = SimpleTestModel()
            client = SatelliteFlowerClient(
                cid=f"sat_{i}",
                model=model,
                train_loader=None,  # 测试不需要实际的数据加载器
                test_loader=None,
                device=device,
                config=sat_config
            )
            client.set_coordinator(coordinator)
            clients.append(client)
    
    return {
        'satellites': satellites,
        'coordinator': coordinator,
        'clients': clients,
        'orbit_calculator': OrbitCalculator(satellites[0])
    }

def test_satellite_visibility(test_environment):
    """测试卫星可见性计算"""
    env = test_environment
    current_time = datetime.now()
    
    # 测试同轨道卫星可见性
    sat1 = env['satellites'][0]
    sat2 = env['satellites'][1]
    
    is_visible = env['orbit_calculator'].calculate_satellite_visibility(
        sat1,
        sat2,
        current_time
    )
    
    print(f"\n卫星可见性测试:")
    print(f"卫星1位置: {env['orbit_calculator']._calculate_satellite_position(current_time)}")
    print(f"卫星2位置: {env['orbit_calculator']._calculate_satellite_position(current_time)}")
    print(f"可见性: {'可见' if is_visible else '不可见'}")
    
    assert isinstance(is_visible, bool)

def test_adaptive_scheduling(test_environment):
    """测试自适应调度"""
    env = test_environment
    start_time = datetime.now()
    
    # 模拟资源状态
    resource_states = {
        'sat_0': {'battery': 90, 'memory': 30, 'cpu': 20},
        'sat_1': {'battery': 60, 'memory': 50, 'cpu': 40},
        'sat_2': {'battery': 30, 'memory': 70, 'cpu': 80}
    }
    
    # 模拟任务优先级
    task_priorities = {
        'sat_0': 3,  # 高优先级
        'sat_1': 2,  # 中优先级
        'sat_2': 1   # 低优先级
    }
    
    # 生成调度计划
    schedule = env['orbit_calculator'].adaptive_schedule(
        satellites=env['satellites'],
        start_time=start_time,
        resource_states=resource_states,
        task_priorities=task_priorities,
        duration_hours=2
    )
    
    print("\n调度计划测试:")
    for sat_id, events in schedule.items():
        print(f"\n卫星 {sat_id} 的调度计划:")
        for event in events[:5]:  # 只打印前5个事件
            print(f"- 时间: {event['time'].strftime('%H:%M:%S')}")
            print(f"  动作: {event['action']}")
            print(f"  目标: {event['target']}")
            print(f"  优先级: {event['priority']}")
    
    assert len(schedule) == len(env['satellites'])
    for events in schedule.values():
        assert isinstance(events, list)
        if events:
            assert all(isinstance(event, dict) for event in events)
            assert all('time' in event for event in events)
            assert all('action' in event for event in events)
            assert all('target' in event for event in events)
            assert all('priority' in event for event in events)

def test_resource_monitoring(test_environment):
    """测试资源监控和动态调度更新"""
    env = test_environment
    client = env['clients'][0]
    
    # 获取资源状态
    resource_state = client.get_resource_state()
    print("\n资源状态测试:")
    print(f"电池电量: {resource_state['battery']}%")
    print(f"内存使用: {resource_state['memory']}%")
    print(f"CPU使用: {resource_state['cpu']}%")
    
    assert 'battery' in resource_state
    assert 'memory' in resource_state
    assert 'cpu' in resource_state
    
    # 测试调度更新
    client.update_communication_schedule()
    assert client.current_schedule is not None
    
    print("\n当前调度计划:")
    for event in list(client.current_schedule.values())[0][:3]:  # 只打印前3个事件
        print(f"- 时间: {event['time'].strftime('%H:%M:%S')}")
        print(f"  动作: {event['action']}")
        print(f"  目标: {event['target']}")

def test_coordinator_selection(test_environment):
    """测试最佳协调者选择"""
    env = test_environment
    start_time = datetime.now()
    
    best_coordinator, score = env['orbit_calculator'].find_best_coordinator(
        satellites=env['satellites'],
        start_time=start_time,
        duration_hours=24
    )
    
    print("\n协调者选择测试:")
    print(f"最佳协调者: Satellite {best_coordinator}")
    print(f"评分: {score:.2f}")
    
    assert isinstance(best_coordinator, str)
    assert isinstance(score, float)
    assert score > 0

def test_adaptive_scheduling():
    """测试自适应调度"""
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
                sat_id=i,  # 直接在创建时设置卫星ID
                is_coordinator=(i == 0)
            )
            satellites.append(sat_config)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 