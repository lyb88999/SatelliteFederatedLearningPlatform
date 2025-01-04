import pytest
import asyncio
from datetime import datetime
import numpy as np
from typing import Dict, List
from flower.config import SatelliteConfig, GroundStationConfig
from flower.scheduler import Scheduler, Task
from flower.monitor import Monitor
from flower.server import SatelliteServer
from flower.orbit_utils import OrbitCalculator

@pytest.mark.asyncio
async def test_federated_learning():
    """测试联邦学习流程"""
    # 创建测试环境
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建调度器和监控器
    scheduler = Scheduler(orbit_calculator)
    monitor = Monitor()
    
    # 创建卫星
    satellites = []
    for orbit_id in range(2):  # 2个轨道面，每个轨道面2颗卫星
        raan = orbit_id * 180.0  # 轨道面均匀分布
        for sat_id in range(2):
            phase_angle = sat_id * 180.0  # 卫星在轨道内均匀分布
            
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                    eccentricity=0.001,      # 近圆轨道
                    inclination=97.6,        # 太阳同步轨道
                    raan=raan,              # 轨道面的方向
                    arg_perigee=phase_angle, # 卫星在轨道内的位置
                    epoch=datetime.now()
                )
            )
    
    # 创建测试模型
    model = {
        'layer1.weight': np.random.randn(10, 10).astype(np.float32),
        'layer1.bias': np.random.randn(10).astype(np.float32),
    }
    
    # 创建服务器并设置模型
    server = SatelliteServer(scheduler, monitor)
    server.set_model(model)
    
    print("\n开始联邦学习测试")
    print(f"卫星数量: {len(satellites)}")
    print(f"模型参数: {[f'{k}: {v.shape}' for k, v in model.items()]}")
    
    # 模拟本地训练
    local_models = []
    for i in range(2):  # 只使用2颗卫星进行测试
        local_model = {
            k: v + np.random.normal(0, 0.1, v.shape).astype(np.float32)
            for k, v in model.items()
        }
        local_models.append(local_model)
    
    # 测试模型聚合
    aggregated_model = server.aggregate_models(local_models)
    
    # 验证聚合结果
    assert isinstance(aggregated_model, dict)
    assert set(aggregated_model.keys()) == set(model.keys())
    for key in model:
        assert aggregated_model[key].shape == model[key].shape
    
    print("\n模型聚合测试通过")
    
    # 测试任务调度
    task = Task(
        task_id="test_task",
        satellite_id=satellites[0].sat_id,
        start_time=datetime.now(),
        duration=60.0
    )
    scheduler.add_task(task)
    
    scheduled_tasks = scheduler.schedule_tasks()
    assert len(scheduled_tasks) > 0
    
    print("\n任务调度测试通过")
    
    # 测试卫星状态监控
    monitor.update_satellite_status(
        satellite=satellites[0],
        is_active=True,
        link_quality=None
    )
    
    status = monitor.get_satellite_status(satellites[0].sat_id)
    assert status is not None
    assert status.is_active
    
    print("\n卫星监控测试通过")
    
    # 测试训练流程
    await server.start_training(model, satellites[:2])  # 使用卫星配置而不是客户端
    assert server.current_round > 0
    
    print(f"\n完成 {server.current_round} 轮训练")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 