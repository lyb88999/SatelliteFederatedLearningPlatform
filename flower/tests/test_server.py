import pytest
import pytest_asyncio
import asyncio
import websockets
import json
import socket
from datetime import datetime, timedelta
from flower.config import SatelliteConfig, GroundStationConfig
from flower.server import SatelliteServer

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
    
    # 创建服务器
    server_instance = SatelliteServer(config)
    
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

@pytest.mark.asyncio
async def test_client_connection(server):
    """测试客户端连接"""
    try:
        async with websockets.connect(f"ws://localhost:{server.port}") as websocket:
            # 发送注册消息
            await websocket.send(json.dumps({"client_id": "test_client"}))
            
            # 发送状态更新
            status = {
                "type": "status_update",
                "position": [1000.0, 2000.0, 3000.0],
                "velocity": [1.0, 2.0, 3.0],
                "battery_level": 85.5,
                "temperature": 25.3,
                "memory_usage": 45.7
            }
            await websocket.send(json.dumps(status))
            
            # 请求通信窗口
            request = {
                "type": "request_window"
            }
            await websocket.send(json.dumps(request))
            
            # 接收响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                data = json.loads(response)
                
                assert data["type"] == "window_response"
                assert isinstance(data["windows"], list)
            except asyncio.TimeoutError:
                pytest.fail("等待响应超时")
    except Exception as e:
        pytest.fail(f"连接失败: {str(e)}")

@pytest.mark.asyncio
async def test_task_scheduling(server):
    """测试任务调度"""
    try:
        async with websockets.connect(f"ws://localhost:{server.port}") as websocket:
            await websocket.send(json.dumps({"client_id": "test_client"}))
            
            # 添加任务
            task = {
                "type": "add_task",
                "task_id": "test_task",
                "duration": 300,
                "priority": 10,
                "station_id": "Beijing",
                "deadline": (datetime.now() + timedelta(hours=12)).isoformat()
            }
            await websocket.send(json.dumps(task))
            
            # 等待任务被处理
            await asyncio.sleep(0.1)
            
            # 验证任务是否被添加到调度器
            assert len(server.scheduler.tasks) == 1
            assert server.scheduler.tasks[0].task_id == "test_task"
    except Exception as e:
        pytest.fail(f"连接失败: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 