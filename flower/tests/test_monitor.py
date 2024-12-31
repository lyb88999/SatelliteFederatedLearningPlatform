import pytest
from datetime import datetime, timedelta
from flower.config import SatelliteConfig, GroundStationConfig
from flower.monitor import Monitor, LinkQuality, SatelliteStatus
from flower.scheduler import CommunicationWindow

@pytest.fixture
def monitor():
    # 创建测试配置
    ground_stations = [
        GroundStationConfig(
            "Beijing", 
            39.9042, 
            116.4074, 
            coverage_radius=2000,
            min_elevation=10.0
        )
    ]
    
    config = SatelliteConfig(
        orbit_altitude=550.0,
        orbit_inclination=97.6,
        orbital_period=95,
        ground_stations=ground_stations,
        ascending_node=0.0,
        mean_anomaly=0.0
    )
    
    return Monitor(config)

def test_satellite_status_monitoring(monitor):
    """测试卫星状态监控"""
    now = datetime.now()
    
    # 添加一些测试数据
    status = SatelliteStatus(
        timestamp=now,
        position=(1000.0, 2000.0, 3000.0),
        velocity=(1.0, 2.0, 3.0),
        battery_level=85.5,
        temperature=25.3,
        memory_usage=45.7
    )
    
    monitor.update_satellite_status(status)
    health = monitor.get_satellite_health()
    
    assert health["battery_level"] == 85.5
    assert health["temperature"] == 25.3
    assert health["memory_usage"] == 45.7

def test_link_quality_monitoring(monitor):
    """测试通信质量监控"""
    now = datetime.now()
    
    # 添加测试数据
    quality = LinkQuality(
        snr=25.5,
        bit_error_rate=1e-6,
        throughput=100.0
    )
    
    monitor.record_link_quality("Beijing", quality)
    stats = monitor.get_link_statistics(
        "Beijing",
        now - timedelta(minutes=5),
        now + timedelta(minutes=5)
    )
    
    assert stats["avg_snr"] == 25.5
    assert stats["avg_throughput"] == 100.0

def test_communication_window_tracking(monitor):
    """测试通信窗口跟踪"""
    now = datetime.now()
    
    window = CommunicationWindow(
        station_id="Beijing",
        start_time=now,
        end_time=now + timedelta(minutes=10),
        max_elevation=45.0,
        min_distance=800.0
    )
    
    monitor.start_communication_window(window)
    assert monitor.current_window == window
    
    monitor.end_communication_window()
    assert monitor.current_window is None
    assert len(monitor.window_history) == 1

if __name__ == "__main__":
    pytest.main([__file__]) 