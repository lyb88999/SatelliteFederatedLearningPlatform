# server.py
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from .config import SatelliteConfig
from .scheduler import Scheduler, Task
from .monitor import Monitor, LinkQuality
import json
import websockets

class SatelliteServer:
    """卫星服务器"""
    def __init__(self, scheduler: Scheduler, monitor: Monitor, host: str = "localhost", port: int = 8765):
        self.scheduler = scheduler
        self.monitor = monitor
        self.model = None
        self.current_round = 0
        self.host = host
        self.port = port
        self.clients = {}  # 存储连接的客户端
        
    def set_model(self, model: Dict):
        """设置初始模型"""
        self.model = model.copy()  # 创建副本
        
    async def start_training(self, model: Dict, satellites: List[SatelliteConfig]):
        """开始训练"""
        self.model = model
        self.current_round = 0
        
        while self.current_round < 3:  # 将轮次减少到3
            # 创建分发任务
            tasks = []
            for sat in satellites:
                task = Task(
                    task_id=f"round_{self.current_round}_sat_{sat.sat_id}",
                    satellite_id=sat.sat_id,
                    start_time=datetime.now(),
                    duration=60.0  # 1分钟
                )
                tasks.append(task)
                
            # 调度任务
            scheduled_tasks = self.scheduler.schedule_tasks()
            
            # 等待任务完成
            await asyncio.sleep(1.0)
            
            self.current_round += 1
            
    def aggregate_models(self, models: List[Dict]) -> Dict:
        """聚合模型"""
        if not models:
            return self.model.copy() if self.model else {}
            
        aggregated = {}
        for key in self.model.keys():
            params = [m[key] for m in models]
            aggregated[key] = np.mean(params, axis=0)
            
        return aggregated

    async def start_server(self):
        """启动WebSocket服务器"""
        async with websockets.serve(self.handle_connection, self.host, self.port):
            print(f"服务器启动在 ws://{self.host}:{self.port}")
            await asyncio.Future()  # 保持服务器运行
            
    async def handle_connection(self, websocket, path=None):
        """处理WebSocket连接"""
        try:
            client_id = None
            async for message in websocket:
                data = json.loads(message)
                
                if data.get("type") == "register":
                    # 客户端注册
                    client_id = data["client_id"]
                    self.clients[client_id] = websocket
                    await self.send_response(websocket, {
                        "type": "register_response",
                        "status": "success"
                    })
                    
                elif data.get("type") == "status_update":
                    # 更新卫星状态
                    if client_id:
                        self.monitor.update_satellite_status(client_id, data)
                        
                elif data.get("type") == "request_task":
                    # 请求任务
                    if client_id:
                        task = self.scheduler.get_next_task(client_id)
                        await self.send_response(websocket, {
                            "type": "task_response",
                            "task": task.to_dict() if task else None
                        })
                        
        except websockets.exceptions.ConnectionClosed:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
        except Exception as e:
            print(f"连接处理错误: {str(e)}")
            
    async def send_response(self, websocket, data):
        """发送响应"""
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            print(f"发送响应错误: {str(e)}")

async def start_server(model: Dict, satellites: List[SatelliteConfig]):
    """启动服务器"""
    scheduler = Scheduler(None)  # TODO: 添加轨道计算器
    monitor = Monitor()
    server = SatelliteServer(scheduler, monitor)
    await server.start_training(model, satellites)
