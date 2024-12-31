# status_server.py
import asyncio
import websockets
import json
from dataclasses import dataclass, asdict
from typing import Dict, List
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)

@dataclass
class ClientStatus:
    client_id: str
    round: int
    loss: float
    accuracy: float
    parameters_version: int
    training_samples: int
    timestamp: str = ""  # 添加timestamp字段

class FederatedStatusServer:
    def __init__(self):
        self.clients: Dict[str, ClientStatus] = {}
        self.connected_clients = set()
        self.current_round = 0
        
    async def register(self, websocket):
        self.connected_clients.add(websocket)
        logging.info(f"Client connected. Total clients: {len(self.connected_clients)}")
        # 立即发送当前状态
        await self.broadcast_status()
        
    async def unregister(self, websocket):
        self.connected_clients.remove(websocket)
        logging.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")
        
    async def update_status(self, data: dict):
        try:
            client_id = data.get('client_id')
            if client_id:
                # 如果没有timestamp，添加当前时间
                if 'timestamp' not in data:
                    data['timestamp'] = datetime.now().isoformat()
                
                logging.info(f"Received update from client {client_id}: {data}")
                self.clients[client_id] = ClientStatus(**data)
                await self.broadcast_status()
        except Exception as e:
            logging.error(f"Error updating status: {e}")
            logging.error(f"Received data: {data}")
            
    async def broadcast_status(self):
        if not self.connected_clients:
            return
            
        try:
            status = {
                'clients': {cid: asdict(status) for cid, status in self.clients.items()},
                'current_round': self.current_round,
                'total_clients': len(self.clients)
            }
            
            message = json.dumps(status)
            logging.info(f"Broadcasting status: {len(self.clients)} clients")
            await asyncio.gather(
                *[client.send(message) for client in self.connected_clients]
            )
        except Exception as e:
            logging.error(f"Error broadcasting status: {e}")

    async def handler(self, websocket):
        await self.register(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.update_status(data)
        except Exception as e:
            logging.error(f"Error in handler: {e}")
        finally:
            await self.unregister(websocket)

    async def start_server(self, host='localhost', port=8765):
        try:
            async with websockets.serve(self.handler, host, port) as server:
                logging.info(f"Status server starting on ws://{host}:{port}")
                await asyncio.Future()  # run forever
        except Exception as e:
            logging.error(f"Error starting server: {e}")

async def main():
    server = FederatedStatusServer()
    await server.start_server()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server shutting down...")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
