import multiprocessing
import time
from flower.server import start_server
from flower.client import start_client
from flower.config import GroundStationConfig, SatelliteConfig

def run_server():
    # 创建地面站配置
    ground_stations = [
        GroundStationConfig("Beijing", 39.9042, 116.4074, 500),
        GroundStationConfig("Shanghai", 31.2304, 121.4737, 500),
        GroundStationConfig("Guangzhou", 23.1291, 113.2644, 500)
    ]
    
    # 启动服务器
    start_server()

def run_client(cid: int):
    # 创建卫星配置
    config = SatelliteConfig(
        orbit_altitude=550.0,
        orbit_inclination=97.6,
        ground_stations=[
            GroundStationConfig("Beijing", 39.9042, 116.4074, 500),
            GroundStationConfig("Shanghai", 31.2304, 121.4737, 500),
            GroundStationConfig("Guangzhou", 23.1291, 113.2644, 500)
        ]
    )
    
    # 启动客户端
    start_client(cid)

def main():
    # 启动服务器进程
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    
    # 等待服务器启动
    time.sleep(5)
    
    # 启动3个卫星客户端
    client_processes = []
    for i in range(3):
        client_process = multiprocessing.Process(target=run_client, args=(i,))
        client_processes.append(client_process)
        client_process.start()
        time.sleep(2)  # 间隔启动客户端
    
    # 等待所有进程完成
    server_process.join()
    for client_process in client_processes:
        client_process.join()

if __name__ == "__main__":
    main() 