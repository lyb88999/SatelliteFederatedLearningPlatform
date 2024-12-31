import pytest
from datetime import datetime, timedelta
from flower.config import SatelliteConfig, GroundStationConfig
from flower.scheduler import Scheduler, Task, CommunicationWindow

@pytest.fixture
def scheduler():
    # 创建测试用的地面站
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
    
    # 创建卫星配置
    config = SatelliteConfig(
        orbit_altitude=550.0,
        orbit_inclination=97.6,
        orbital_period=95,
        ground_stations=ground_stations,
        ascending_node=0.0,
        mean_anomaly=0.0
    )
    
    return Scheduler(config)

def test_predict_windows(scheduler):
    """测试通信窗口预测"""
    start_time = datetime.now()
    windows = scheduler.predict_windows(start_time, duration_hours=24)
    
    assert len(windows) > 0
    for window in windows:
        assert isinstance(window, CommunicationWindow)
        assert window.end_time > window.start_time
        assert window.max_elevation >= 10.0  # 最小仰角
        assert window.min_distance <= 2000  # 最大覆盖距离

def test_task_scheduling(scheduler):
    """测试任务调度"""
    start_time = datetime.now()
    
    # 添加测试任务
    tasks = [
        Task("task1", 300, 10, "Beijing", start_time + timedelta(hours=12)),
        Task("task2", 180, 8, "Shanghai", start_time + timedelta(hours=6)),
        Task("task3", 240, 5, "Beijing", start_time + timedelta(hours=24))
    ]
    
    for task in tasks:
        scheduler.add_task(task)
    
    # 执行调度
    schedule = scheduler.schedule_tasks(start_time, duration_hours=24)
    
    # 验证调度结果
    assert len(schedule) == 2  # 两个地面站
    assert all(isinstance(station_schedule, list) for station_schedule in schedule.values())
    
    # 检查任务分配是否合理
    for station_id, tasks in schedule.items():
        for task, scheduled_time in tasks:
            assert scheduled_time >= start_time
            assert scheduled_time <= start_time + timedelta(hours=24)
            assert task.station_id == station_id

def test_window_metrics(scheduler):
    """测试通信窗��的指标计算"""
    start_time = datetime.now()
    windows = scheduler.predict_windows(start_time, duration_hours=24)
    
    for window in windows:
        # 检查指标的合理性
        assert 0 <= window.max_elevation <= 90  # 仰角范围
        assert 550 <= window.min_distance <= 2000  # 距离范围（km）
        
        # 检查时间间隔的合理性
        duration = (window.end_time - window.start_time).total_seconds() / 60
        assert 5 <= duration <= 15  # 典型的LEO卫星通过时间为5-15分钟

if __name__ == "__main__":
    pytest.main([__file__]) 