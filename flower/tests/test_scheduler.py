import pytest
from datetime import datetime
from flower.config import SatelliteConfig, GroundStationConfig
from flower.scheduler import Scheduler, Task
from flower.orbit_utils import OrbitCalculator

@pytest.fixture
def scheduler():
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建调度器
    scheduler = Scheduler(orbit_calculator)
    
    # 创建测试卫星
    satellite = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    # 创建测试任务
    task = Task(
        task_id="test_task",
        satellite_id=satellite.sat_id,
        start_time=datetime.now(),
        duration=60.0
    )
    scheduler.add_task(task)
    
    return scheduler

def test_task_scheduling(scheduler):
    """测试任务调度"""
    tasks = scheduler.schedule_tasks()
    assert len(tasks) > 0

if __name__ == "__main__":
    pytest.main([__file__]) 