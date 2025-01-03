import pytest
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator
from flower.client import SatelliteFlowerClient, OrbitCoordinatorClient
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class SimpleTestModel(nn.Module):
    """用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)

def create_ground_stations() -> List[GroundStationConfig]:
    """创建全球范围内的地面站网络"""
    return [
        GroundStationConfig("Beijing", 39.9042, 116.4074, 2000, 10.0),
        GroundStationConfig("Shanghai", 31.2304, 121.4737, 2000, 10.0),
        GroundStationConfig("Tokyo", 35.6762, 139.6503, 2000, 10.0),
        GroundStationConfig("Singapore", 1.3521, 103.8198, 2000, 10.0),
        GroundStationConfig("Moscow", 55.7558, 37.6173, 2000, 10.0),
        GroundStationConfig("London", 51.5074, -0.1278, 2000, 10.0),
        GroundStationConfig("New York", 40.7128, -74.0060, 2000, 10.0),
        GroundStationConfig("Los Angeles", 34.0522, -118.2437, 2000, 10.0),
        GroundStationConfig("Sydney", -33.8688, 151.2093, 2000, 10.0),
        GroundStationConfig("Cape Town", -33.9249, 18.4241, 2000, 10.0)
    ]

def create_multi_orbit_satellites(num_orbits: int = 3, sats_per_orbit: int = 4) -> List[SatelliteConfig]:
    """创建多轨道卫星网络"""
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    satellites = []
    
    for orbit_id in range(num_orbits):
        # 每个轨道略微不同的参数
        altitude = 550.0 + orbit_id * 50  # 轨道高度递增
        
        for sat_idx in range(sats_per_orbit):
            # 在轨道内均匀分布卫星
            phase_angle = (360.0 / sats_per_orbit) * sat_idx
            
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + altitude,
                    eccentricity=0.001,
                    inclination=97.6 + orbit_id * 2,  # 倾角递增
                    raan=orbit_id * 120.0,  # 轨道面均匀分布
                    arg_perigee=phase_angle,
                    epoch=datetime.now()
                )
            )
    
    return satellites

@pytest.fixture
def large_scale_environment():
    """创建大规模测试环境"""
    satellites = create_multi_orbit_satellites()
    device = torch.device("cpu")
    
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
    for orbit_id, orbit_sats in satellites_by_orbit.items():
        orbit_clients = []
        orbit_coordinator = None
        
        for i, sat_config in enumerate(orbit_sats):
            model = SimpleTestModel()
            
            if sat_config.is_coordinator:
                # 创建协调者
                coordinator = OrbitCoordinatorClient(
                    cid=f"orbit_{orbit_id}_sat_{i}",
                    model=model,
                    device=device,
                    config=sat_config
                )
                orbit_coordinator = coordinator
                coordinators.append(coordinator)
            else:
                # 创建普通客户端
                client = SatelliteFlowerClient(
                    cid=f"orbit_{orbit_id}_sat_{i}",
                    model=model,
                    train_loader=None,
                    test_loader=None,
                    device=device,
                    config=sat_config
                )
                orbit_clients.append(client)
                clients.append(client)
        
        # 设置轨道内客户端的协调者
        for client in orbit_clients:
            client.set_coordinator(orbit_coordinator)
    
    return {
        'satellites': satellites,
        'coordinators': coordinators,
        'clients': clients,
        'orbit_calculator': OrbitCalculator(satellites[0])
    }

def test_multi_orbit_visibility(large_scale_environment):
    """测试多轨道卫星可见性"""
    env = large_scale_environment
    current_time = datetime.now()
    
    # 测试不同轨道间的可见性
    visibility_matrix = {}
    for sat1 in env['satellites']:
        sat1_id = f"orbit_{sat1.orbit_id}"
        visibility_matrix[sat1_id] = {}
        
        for sat2 in env['satellites']:
            sat2_id = f"orbit_{sat2.orbit_id}"
            
            is_visible = env['orbit_calculator'].check_satellite_visibility(
                sat1,
                sat2,
                current_time
            )
            visibility_matrix[sat1_id][sat2_id] = is_visible
    
    print("\n" + "="*50)
    print("轨道可见性矩阵:")
    print("="*50)
    # 打印表头
    headers = sorted(visibility_matrix.keys())
    print(f"{'轨道':^10}", end="")
    for header in headers:
        print(f"{header:^12}", end="")
    print("\n" + "-"*70)
    
    # 打印矩阵
    for orbit1 in headers:
        print(f"{orbit1:^10}", end="")
        for orbit2 in headers:
            visible = "✓" if visibility_matrix[orbit1][orbit2] else "✗"
            print(f"{visible:^12}", end="")
        print()
    print("="*70 + "\n")
    
    # 验证同轨道卫星可见性
    for sat1 in env['satellites']:
        for sat2 in env['satellites']:
            if sat1.orbit_id == sat2.orbit_id and sat1 != sat2:
                assert env['orbit_calculator'].check_satellite_visibility(
                    sat1,
                    sat2,
                    current_time
                ), f"同轨道卫星 {sat1.orbit_id} 应该可见"
    
    # 添加可视化
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制地球
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = env['orbit_calculator'].earth_radius * np.outer(np.cos(u), np.sin(v))
    y = env['orbit_calculator'].earth_radius * np.outer(np.sin(u), np.sin(v))
    z = env['orbit_calculator'].earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    
    # 绘制卫星和轨道
    colors = ['r', 'g', 'b']
    markers = ['o', 's', '^']  # 不同形状的标记
    
    # 按轨道组织卫星
    satellites_by_orbit = {}
    for sat in env['satellites']:
        if sat.orbit_id not in satellites_by_orbit:
            satellites_by_orbit[sat.orbit_id] = []
        satellites_by_orbit[sat.orbit_id].append(sat)
    
    # 为每个轨道绘制卫星
    for orbit_id, sats in satellites_by_orbit.items():
        color = colors[orbit_id % len(colors)]
        marker = markers[orbit_id % len(markers)]
        
        # 绘制轨道路径
        orbit_points = []
        for sat in sats:
            env['orbit_calculator'].set_current_satellite(sat)  # 设置当前卫星
            for t in range(0, 360, 5):
                time = current_time + timedelta(minutes=t)
                pos = env['orbit_calculator']._calculate_satellite_position(time)
                orbit_points.append(pos)
        orbit_points = np.array(orbit_points)
        ax.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], 
                '--', c=color, alpha=0.3)
        
        # 绘制卫星
        for sat in sats:
            pos = env['orbit_calculator']._calculate_satellite_position(current_time)
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=color, marker=marker, s=100,
                      label=f'Orbit {orbit_id}' if sat == sats[0] else "")
            
            # 绘制可见性连线
            for other_sat in sats:
                if sat != other_sat and env['orbit_calculator'].check_satellite_visibility(
                    sat, other_sat, current_time):
                    other_pos = env['orbit_calculator']._calculate_satellite_position(current_time)
                    ax.plot([pos[0], other_pos[0]], 
                           [pos[1], other_pos[1]], 
                           [pos[2], other_pos[2]], 
                           c=color, alpha=0.3)
    
    # 绘制地面站
    for station in env['satellites'][0].ground_stations:
        pos = env['orbit_calculator']._calculate_ground_station_position(station, current_time)
        ax.scatter(pos[0], pos[1], pos[2], 
                  c='k', marker='*', s=200,
                  label=f'Ground Station: {station.station_id}')
    
    ax.set_title('卫星轨道和可见性')
    ax.legend()
    plt.show()

def test_large_scale_scheduling(large_scale_environment):
    """测试大规模调度"""
    env = large_scale_environment
    start_time = datetime.now()
    
    # 模拟所有卫星的资源状态
    resource_states = {}
    task_priorities = {}
    
    # 为每个卫星生成随机但合理的资源状态和优先级
    for i, sat in enumerate(env['satellites']):
        sat_id = f"orbit_{sat.orbit_id}_sat_{i}"
        
        # 根据轨道位置设置不同的资源状态
        orbit_factor = (sat.orbit_id + 1) / len(set(s.orbit_id for s in env['satellites']))
        resource_states[sat_id] = {
            'battery': np.random.uniform(60, 100) * orbit_factor,
            'memory': np.random.uniform(20, 80) * orbit_factor,
            'cpu': np.random.uniform(10, 70) * orbit_factor
        }
        
        # 协调者获得更高优先级
        task_priorities[sat_id] = 3 if sat.is_coordinator else np.random.randint(1, 3)
    
    # 生成调度计划
    schedule = env['orbit_calculator'].schedule_adaptive(
        satellites=env['satellites'],
        start_time=start_time,
        resource_states=resource_states,
        task_priorities=task_priorities,
        duration_hours=2
    )
    
    print("\n" + "="*50)
    print("大规模调度测试:")
    print("="*50)
    
    # 按轨道分组显示调度计划
    satellites_by_orbit = {}
    for sat_id in schedule.keys():
        orbit_id = sat_id.split('_')[1]
        if orbit_id not in satellites_by_orbit:
            satellites_by_orbit[orbit_id] = []
        satellites_by_orbit[orbit_id].append(sat_id)
    
    for orbit_id, sat_ids in satellites_by_orbit.items():
        print(f"\n轨道 {orbit_id} 的调度计划:")
        print("-" * 40)
        
        for sat_id in sorted(sat_ids):
            events = schedule[sat_id]
            print(f"\n卫星 {sat_id}:")
            print(f"资源状态: {resource_states[sat_id]}")
            print(f"任务优先级: {task_priorities[sat_id]}")
            print("通信计划:")
            
            for event in events[:3]:  # 只打印前3个事件
                print(f"- 时间: {event['time'].strftime('%H:%M:%S')}")
                print(f"  动作: {event['action']}")
                print(f"  目标: {event['target']}")
                print(f"  优先级: {event['priority']}")
                print(f"  持续时间: {event['duration'].total_seconds()/60:.1f}分钟")
    
    print("\n" + "="*50)
    
    # 验证调度计划
    assert len(schedule) == len(env['satellites'])
    for events in schedule.values():
        if events:
            assert all('time' in event for event in events)
            assert all('action' in event for event in events)
            assert all('target' in event for event in events)
            assert all('priority' in event for event in events)

def test_coordinator_network(large_scale_environment):
    """测试协调者网络"""
    env = large_scale_environment
    start_time = datetime.now()
    
    # 测试每个轨道的协调者选择
    coordinator_scores = env['orbit_calculator'].select_best_coordinator(
        satellites=env['satellites'],
        start_time=start_time,
        duration_hours=24
    )
    
    print("\n" + "="*50)
    print("轨道协调者选择结果:")
    print("="*50)
    
    for orbit_id, (coordinator_id, score) in coordinator_scores.items():
        print(f"\n轨道 {orbit_id}:")
        print("-" * 30)
        print(f"最佳协调者: {coordinator_id}")
        print(f"评分: {score:.2f}")
        
        # 获取该轨道的所有卫星
        orbit_sats = [sat for sat in env['satellites'] if sat.orbit_id == int(orbit_id)]
        print(f"轨道内卫星数量: {len(orbit_sats)}")
        
        # 显示与地面站的可见性
        ground_station_visibility = 0
        for station in orbit_sats[0].ground_stations:
            for minutes in range(0, 24 * 60, 10):
                check_time = start_time + timedelta(minutes=minutes)
                if env['orbit_calculator'].calculate_visibility(station, check_time):
                    ground_station_visibility += 1
        print(f"地面站可见性窗口数: {ground_station_visibility}")
    
    print("\n" + "="*50)
    
    # 验证结果
    assert len(coordinator_scores) == len(set(sat.orbit_id for sat in env['satellites']))
    for orbit_id, (coordinator_id, score) in coordinator_scores.items():
        assert isinstance(coordinator_id, str)
        assert isinstance(score, float)
        assert score > 0
        # 验证选择的协调者确实属于该轨道
        assert coordinator_id.startswith(f"orbit_{orbit_id}_sat_")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 