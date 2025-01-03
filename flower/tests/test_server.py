import pytest
import pytest_asyncio
import asyncio
import websockets
import json
import socket
from datetime import datetime, timedelta
from flower.config import SatelliteConfig, GroundStationConfig
from flower.server import SatelliteServer
from flower.orbit_utils import OrbitCalculator
from flower.scheduler import Scheduler
from flower.monitor import Monitor

def find_free_port():
    """找到一个可用的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

@pytest_asyncio.fixture
async def server():
    # 创建测试配置
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建调度器和监控器
    scheduler = Scheduler(orbit_calculator)
    monitor = Monitor()
    
    ground_stations = [
        GroundStationConfig(
            station_id="Beijing",
            latitude=39.9042,
            longitude=116.4074,
            max_range=2000,
            min_elevation=10.0,
            max_satellites=4
        )
    ]
    
    config = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,
        eccentricity=0.001,
        inclination=97.6,
        raan=0.0,
        arg_perigee=0.0,
        epoch=datetime.now()
    )
    
    # 创建服务器
    server_instance = SatelliteServer(scheduler, monitor)
    
    # 找到可用端口
    port = find_free_port()
    
    # 启动服务器
    server_task = None
    server_started = asyncio.Event()
    
    async def start_server():
        nonlocal server_task
        ws_server = await websockets.serve(
            server_instance.handle_connection, 
            "localhost", 
            port
        )
        server_started.set()
        await ws_server.wait_closed()
    
    try:
        server_task = asyncio.create_task(start_server())
        await asyncio.wait_for(server_started.wait(), timeout=5.0)
        server_instance.port = port  # 保存端口号以供测试使用
        yield server_instance
    finally:
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

@pytest.mark.skip(reason="WebSocket service not implemented yet")
@pytest.mark.asyncio
async def test_client_connection(server):
    """测试客户端连接"""
    pass

@pytest.mark.skip(reason="WebSocket service not implemented yet")
@pytest.mark.asyncio
async def test_task_scheduling(server):
    """测试任务调度"""
    pass

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 