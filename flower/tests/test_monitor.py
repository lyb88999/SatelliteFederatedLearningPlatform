import pytest
from datetime import datetime
from flower.config import SatelliteConfig, GroundStationConfig
from flower.monitor import Monitor, LinkQuality
from flower.orbit_utils import OrbitCalculator

@pytest.fixture
def monitor():
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建测试卫星
    satellite = SatelliteConfig(
        orbit_id=0,
        sat_id="0",
        semi_major_axis=earth_radius + 550.0,
        eccentricity=0.001,
        inclination=98.0,
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    # 创建监控器
    monitor = Monitor()
    monitor.update_satellite_status(
        satellite=satellite,
        is_active=True,
        link_quality=LinkQuality(
            signal_strength=-70.0,
            bit_error_rate=1e-6,
            latency=50.0
        )
    )
    
    return monitor

def test_satellite_status_monitoring(monitor):
    """测试卫星状态监控"""
    status = monitor.get_satellite_status("0")
    assert status is not None
    assert status.is_active
    assert status.link_quality is not None 