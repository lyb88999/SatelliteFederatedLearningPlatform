from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import (
    Metrics, 
    FitRes, 
    Status, 
    Code,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns, 
    EvaluateIns,
    FitRes,
    EvaluateRes,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from datetime import datetime
from .orbit_utils import OrbitCalculator
from .config import SatelliteConfig
from .ground_station import GroundStation
from .client import SatelliteFlowerClient
import time
import random
import torch
from collections import OrderedDict
import numpy as np
import asyncio
import json
from .visualization import FederatedLearningVisualizer

class SatelliteFedAvg(FedAvg):
    """分层联邦平均策略"""
    
    def aggregate_orbit(self, orbit_id: int, results: List[Tuple[SatelliteFlowerClient, Tuple[int, Dict[str, float]]]]) -> Optional[Parameters]:
        """轨道内聚合"""
        if not results:
            return None
            
        # 基于性能的权重计算
        weights_results = []
        for client, (num_examples, metrics) in results:
            # 获取当前模型参数
            params = [val.cpu().numpy() for _, val in client.model.state_dict().items()]
            
            # 计算权重
            loss = metrics.get("loss", float('inf'))
            accuracy = metrics.get("accuracy", 0.0)
            # 同时考虑loss和accuracy
            weight = (1.0 / (loss + 1e-10)) * (1.0 + accuracy)
            
            weights_results.append(
                (params, weight)
            )
        
        if not weights_results:
            return None
            
        # 归一化权重
        total_weight = sum(w for _, w in weights_results)
        aggregated_params = [np.zeros_like(param) for param in weights_results[0][0]]
        
        for parameters, weight in weights_results:
            weight = weight / total_weight
            for i, param in enumerate(parameters):
                aggregated_params[i] += param * weight
                
        print(f"轨道 {orbit_id} 内聚合完成，参与节点数: {len(weights_results)}")
        return ndarrays_to_parameters(aggregated_params)
        
    def aggregate_ground_station(self, station_id: str, orbit_models: List[Tuple[int, Parameters]]) -> Optional[Parameters]:
        """地面站级聚合"""
        if not orbit_models:
            return None
            
        # 简单平均所有轨道的模型
        params_list = [parameters_to_ndarrays(params) for _, params in orbit_models]
        aggregated_params = []
        
        for param_idx in range(len(params_list[0])):
            param_sum = sum(params[param_idx] for params in params_list)
            aggregated_params.append(param_sum / len(params_list))
            
        print(f"地面站 {station_id} 聚合了 {len(orbit_models)} 个轨道的模型")
        return ndarrays_to_parameters(aggregated_params)
        
    def aggregate_global(self, station_models: List[Tuple[str, Parameters]]) -> Optional[Parameters]:
        """全局聚合"""
        if not station_models:
            return None
            
        # 简单平均所有地面站的模型
        params_list = [parameters_to_ndarrays(params) for _, params in station_models]
        aggregated_params = []
        
        for param_idx in range(len(params_list[0])):
            param_sum = sum(params[param_idx] for params in params_list)
            aggregated_params.append(param_sum / len(params_list))
            
        return ndarrays_to_parameters(aggregated_params)

class SatelliteFlowerServer:
    """卫星联邦学习服务器"""
    def __init__(self, 
                 satellites: List[SatelliteConfig],
                 ground_stations: List[GroundStation],
                 orbit_calculator: OrbitCalculator,
                 strategy: Optional[fl.server.strategy.Strategy] = None,
                 num_rounds: int = 5):
        """初始化服务器
        
        Args:
            satellites: 卫星列表
            ground_stations: 地面站列表
            orbit_calculator: 轨道计算器
            strategy: 联邦学习策略，默认使用FedAvg
            num_rounds: 训练轮数
        """
        self.satellites = satellites
        self.ground_stations = ground_stations
        self.orbit_calculator = orbit_calculator
        self.strategy = strategy or SatelliteFedAvg()
        self.current_round = 0
        self.model = None
        self.client_manager = fl.server.client_manager.SimpleClientManager()
        self.num_rounds = num_rounds
        self.best_accuracy = 0.0
        self.patience = 3
        self.no_improve_count = 0
        self.visualizer = FederatedLearningVisualizer()
        
    async def train_round(self, clients: List[SatelliteFlowerClient]) -> Dict:
        """执行一轮分层训练"""
        self.current_round += 1
        print(f"\n{'='*20} 轮次 {self.current_round}/{self.num_rounds} {'='*20}")
        
        # 1. 轨道内聚合
        orbit_models = {}
        orbit_metrics = {}  # 存储每个轨道的指标
        
        for orbit_id in range(6):
            orbit_clients = [c for c in clients if c.satellite_id // 11 == orbit_id]
            if not orbit_clients:
                continue
            
            print(f"\n开始轨道 {orbit_id} 的训练...")
            
            # 训练
            train_tasks = [client.train() for client in orbit_clients]
            train_results = await asyncio.gather(*train_tasks)
            
            # 更新卫星级别的指标
            for client, (num_examples, metrics) in zip(orbit_clients, train_results):
                self.visualizer.update_satellite_metrics(
                    self.current_round,
                    client.satellite_id,
                    metrics
                )
            
            # 轨道内聚合
            orbit_model = self.strategy.aggregate_orbit(
                orbit_id,
                [(client, result) for client, result in zip(orbit_clients, train_results)]
            )
            if orbit_model:
                orbit_models[orbit_id] = orbit_model
                
                # 评估轨道内性能
                eval_tasks = [client.evaluate() for client in orbit_clients]
                eval_results = await asyncio.gather(*eval_tasks)
                orbit_metrics[orbit_id] = self.aggregate_metrics(eval_results)
                
                # 更新轨道级别的指标
                self.visualizer.update_orbit_metrics(
                    self.current_round,
                    orbit_id,
                    orbit_metrics[orbit_id]
                )
        
        # 2. 地面站级聚合
        station_models = []
        current_time = datetime.now()
        
        print("\n开始地面站级聚合...")
        print(f"调试模式: {self.orbit_calculator.debug_mode}")
        print(f"可用轨道: {list(orbit_models.keys())}")
        print(f"地面站数量: {len(self.ground_stations)}")
        
        for station in self.ground_stations:
            visible_orbits = []
            print(f"\n检查地面站 {station.config.station_id} 的可见性...")
            
            for orbit_id, model in orbit_models.items():
                coordinator = self._get_orbit_coordinator(orbit_id)
                if coordinator:
                    is_visible = self.orbit_calculator.check_satellite_visibility(
                        coordinator, station, current_time)
                    print(f"轨道 {orbit_id} {'可见' if is_visible else '不可见'}")
                    
                    if is_visible:
                        visible_orbits.append((orbit_id, model))
                        print(f"地面站 {station.config.station_id} 可见轨道 {orbit_id}")
            
            if visible_orbits:
                station_model = self.strategy.aggregate_ground_station(
                    station.config.station_id, visible_orbits)
                if station_model:
                    station_models.append((station.config.station_id, station_model))
                    print(f"地面站 {station.config.station_id} 完成轨道间聚合，"
                          f"聚合了 {len(visible_orbits)} 个轨道的模型")
        
        print(f"\n地面站级聚合完成，参与地面站数: {len(station_models)}")
        
        # 3. 全局聚合
        if station_models:
            print(f"\n开始全局聚合，参与地面站数: {len(station_models)}")
            self.model = self.strategy.aggregate_global(station_models)
            for client in clients:
                await client.set_model(self.model)  # 更新所有客户端的模型
            print("全局聚合完成")
        
        # 4. 评估
        print("\n开始全局评估...")
        eval_tasks = [client.evaluate() for client in clients]
        eval_results = await asyncio.gather(*eval_tasks)
        
        # 5. 计算全局指标
        aggregated_metrics = self.aggregate_metrics(eval_results)
        print(f"\n本轮训练结果:")
        print(f"- 准确率: {aggregated_metrics['accuracy']:.4f}")
        print(f"- 损失: {aggregated_metrics['loss']:.4f}")
        print(f"- 参与轨道数: {len(orbit_models)}")
        print(f"- 参与地面站数: {len(station_models)}")
        
        # 早停检查
        if aggregated_metrics['accuracy'] > self.best_accuracy:
            self.best_accuracy = aggregated_metrics['accuracy']
            self.no_improve_count = 0
        else:
            self.no_improve_count += 1
            
        if self.no_improve_count >= self.patience:
            print(f"Early stopping triggered after {self.current_round} rounds")
            return aggregated_metrics
        
        # 更新全局指标
        self.visualizer.update_global_metrics(aggregated_metrics)
        
        # 每轮结束时绘制图表
        self.visualizer.plot_satellite_metrics()
        self.visualizer.plot_orbit_metrics()
        self.visualizer.plot_global_metrics()
        
        return aggregated_metrics
        
    def _get_orbit_coordinator(self, orbit_id: int) -> Optional[SatelliteConfig]:
        """获取轨道协调者"""
        orbit_satellites = [sat for sat in self.satellites if sat.orbit_id == orbit_id]
        if not orbit_satellites:
            return None
        # 简单策略：选择第一颗卫星作为协调者
        return orbit_satellites[0]
        
    def aggregate_metrics(self, metrics_list: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """聚合评估指标"""
        if not metrics_list:
            return {
                'accuracy': 0.0,
                'loss': float('inf')
            }
        
        total_examples = sum(num_examples for num_examples, _ in metrics_list)
        if total_examples == 0:
            return {
                'accuracy': 0.0,
                'loss': float('inf')
            }
        
        weighted_accuracy = 0.0
        weighted_loss = 0.0
        
        for num_examples, metrics in metrics_list:
            if num_examples > 0:  # 只聚合有效的指标
                weight = num_examples / total_examples
                weighted_accuracy += metrics.get('accuracy', 0.0) * weight
                weighted_loss += metrics.get('loss', float('inf')) * weight
        
        return {
            'accuracy': weighted_accuracy,
            'loss': weighted_loss
        }