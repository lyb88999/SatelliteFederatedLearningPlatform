# server.py
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
import websockets
import json
from .config import SatelliteConfig, GroundStationConfig
from .scheduler import Scheduler, Task
from .monitor import Monitor, SatelliteStatus, LinkQuality

class SatelliteServer:
    def __init__(self, config: SatelliteConfig):
        self.config = config
        self.scheduler = Scheduler(config)
        self.monitor = Monitor(config)
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        
    async def start(self, host: str = "localhost", port: int = 8765):
        """启动服务器"""
        server = await websockets.serve(self.handle_connection, host, port)
        print(f"服务器启动在 ws://{host}:{port}")
        await server.wait_closed()
        
    async def handle_connection(self, websocket):
        """处理新的连接"""
        try:
            # 等待客户端注册
            message = await websocket.recv()
            data = json.loads(message)
            client_id = data.get("client_id")
            
            if not client_id:
                await websocket.close(1002, "需要client_id")
                return
                
            # 注册客户端
            self.clients[client_id] = websocket
            print(f"客户端 {client_id} 已连接")
            
            try:
                async for message in websocket:
                    await self.handle_message(client_id, message)
            finally:
                # 客户端断开连接
                del self.clients[client_id]
                print(f"客户端 {client_id} 已断开连接")
                
        except Exception as e:
            print(f"连接错误: {e}")
            
    async def handle_message(self, client_id: str, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "status_update":
                # 更新卫星状态
                status = SatelliteStatus(
                    timestamp=datetime.now(),
                    position=tuple(data["position"]),
                    velocity=tuple(data["velocity"]),
                    battery_level=data["battery_level"],
                    temperature=data["temperature"],
                    memory_usage=data["memory_usage"]
                )
                self.monitor.update_satellite_status(status)
                
            elif message_type == "request_window":
                # 请求通信窗口
                windows = self.scheduler.predict_windows(
                    start_time=datetime.now(),
                    duration_hours=24
                )
                response = {
                    "type": "window_response",
                    "windows": [
                        {
                            "station_id": w.station_id,
                            "start_time": w.start_time.isoformat(),
                            "end_time": w.end_time.isoformat(),
                            "max_elevation": w.max_elevation,
                            "min_distance": w.min_distance
                        }
                        for w in windows
                    ]
                }
                await self.clients[client_id].send(json.dumps(response))
                
            elif message_type == "link_quality":
                # 记录链路质量
                quality = LinkQuality(
                    snr=data["snr"],
                    bit_error_rate=data["bit_error_rate"],
                    throughput=data["throughput"]
                )
                self.monitor.record_link_quality(data["station_id"], quality)
                
            elif message_type == "add_task":
                # 添加新任务
                task = Task(
                    task_id=data["task_id"],
                    duration=data["duration"],
                    priority=data["priority"],
                    station_id=data["station_id"],
                    deadline=datetime.fromisoformat(data["deadline"])
                )
                self.scheduler.add_task(task)
                
        except Exception as e:
            print(f"处理消息错误: {e}")
            
    async def broadcast_status(self):
        """广播系统状态"""
        while True:
            try:
                status = {
                    "type": "system_status",
                    "time": datetime.now().isoformat(),
                    "active_clients": len(self.clients),
                    "satellite_health": self.monitor.get_satellite_health()
                }
                
                # 向所有客户端广播
                for client in self.clients.values():
                    try:
                        await client.send(json.dumps(status))
                    except:
                        continue
                        
            except Exception as e:
                print(f"广播错误: {e}")
                
            await asyncio.sleep(5)  # 每5秒广播一次
