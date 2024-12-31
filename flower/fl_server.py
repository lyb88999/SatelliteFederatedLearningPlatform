import flwr as fl
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime
from .server import SatelliteServer
from .config import SatelliteConfig, GroundStationConfig
import asyncio

class SatelliteFederatedServer:
    def __init__(self, satellite_server: SatelliteServer, min_clients: int = 2):
        self.satellite_server = satellite_server
        self.min_clients = min_clients
        
    def start_server(self):
        # 定义策略
        def get_initial_parameters(config):
            """获取初始模型参数"""
            print("正在初始化模型参数...")
            import torch
            from .client import Net
            
            # 创建模型实例
            model = Net()
            
            # 获取模型参数
            weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
            return fl.common.ndarrays_to_parameters(weights)

        # 定义自定义的聚合策略
        class CustomFedAvg(fl.server.strategy.FedAvg):
            def aggregate_fit(self, server_round, results, failures):
                """允许部分客户端失败"""
                if not results:
                    return None, {}
                    
                # 过滤掉返回 None 的结果
                valid_results = [(weights, fit_res) for weights, fit_res in results if fit_res.parameters is not None]
                
                if not valid_results:
                    return None, {}
                    
                return super().aggregate_fit(server_round, valid_results, failures)

            def aggregate_evaluate(self, server_round, results, failures):
                """允许部分客户端失败"""
                if not results:
                    return None, {}
                    
                # 过滤掉返回 None 的结果
                valid_results = [(weights, eval_res) for weights, eval_res in results if eval_res.loss is not None]
                
                if not valid_results:
                    return None, {}
                    
                return super().aggregate_evaluate(server_round, valid_results, failures)
        
        strategy = CustomFedAvg(
            fraction_fit=0.5,
            fraction_evaluate=0.5,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
            initial_parameters=get_initial_parameters({}),
            on_fit_config_fn=lambda _: {
                "epochs": 1,
                "batch_size": 64,
                "timeout": 600
            },
            on_evaluate_config_fn=lambda _: {
                "timeout": 600
            }
        )
        
        # 使用新的 API
        print("启动服务器...")
        
        # 创建客户端管理器
        client_manager = fl.server.SimpleClientManager()
        
        # 创建服务器实例
        server = fl.server.Server(
            client_manager=client_manager,
            strategy=strategy
        )
        
        # 配置服务器
        config = fl.server.ServerConfig(
            num_rounds=3,
            round_timeout=600.0
        )
        
        # 启动服务器
        fl.server.app.start_server(
            server_address="0.0.0.0:8080",
            server=server,
            config=config,
            grpc_max_message_length=1024*1024*1024  # 1GB
        )

def main():
    # 创建地面站配置
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
        ),
        GroundStationConfig(
            "Guangzhou", 
            23.1291, 
            113.2644, 
            coverage_radius=2000,
            min_elevation=10.0
        )
    ]
    
    # 创建卫星配置
    satellite_config = SatelliteConfig(
        orbit_altitude=550.0,
        orbit_inclination=97.6,
        orbital_period=95,
        ground_stations=ground_stations,
        ascending_node=0.0,
        mean_anomaly=0.0,
        orbit_id=0,           # 服务器端的轨道ID设为0
        is_coordinator=False  # 服务器端不是协调者
    )
    
    # 创建卫星服务器
    satellite_server = SatelliteServer(satellite_config)
    
    # 创建联邦学习服务器
    fl_server = SatelliteFederatedServer(satellite_server)
    
    # 启动服务器
    fl_server.start_server()

if __name__ == "__main__":
    main() 