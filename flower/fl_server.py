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
from .config import SatelliteConfig, GroundStationConfig
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
    
    def __init__(self, visualizer: FederatedLearningVisualizer = None):
        super().__init__()
        self.visualizer = visualizer
        self.current_round = 0
        
    async def train_orbit(self, orbit_id: int, orbit_clients: List[SatelliteFlowerClient]):
        """轨道内训练基础实现"""
        print(f"\n开始轨道 {orbit_id} 的训练...")
        
        try:
            # 训练
            train_tasks = [client.train() for client in orbit_clients]
            train_results = await asyncio.gather(*train_tasks)
            
            # 更新卫星级别的指标
            for client, (num_examples, metrics) in zip(orbit_clients, train_results):
                if self.visualizer:
                    print(f"更新卫星 {client.satellite_id} 的训练指标: {metrics}")
                    self.visualizer.update_satellite_metrics(
                        round=self.current_round,
                        satellite_id=client.satellite_id,
                        metrics=metrics,
                        is_training=True
                    )
            
            # 轨道内聚合
            orbit_model = self.aggregate_orbit(
                orbit_id,
                [(client, result) for client, result in zip(orbit_clients, train_results)]
            )
            
            if orbit_model:
                # 更新模型
                for client in orbit_clients:
                    await client.set_model(orbit_model)
                
                # 评估
                eval_tasks = [client.evaluate() for client in orbit_clients]
                eval_results = await asyncio.gather(*eval_tasks)
                
                # 更新评估指标
                for client, (num_examples, metrics) in zip(orbit_clients, eval_results):
                    if self.visualizer:
                        print(f"更新卫星 {client.satellite_id} 的评估指标: {metrics}")
                        self.visualizer.update_satellite_metrics(
                            round=self.current_round,
                            satellite_id=client.satellite_id,
                            metrics=metrics,
                            is_training=False
                        )
                
                # 更新轨道级别的指标
                orbit_metrics = self.aggregate_metrics(eval_results)
                if self.visualizer:
                    self.visualizer.update_orbit_metrics(
                        self.current_round,
                        orbit_id,
                        orbit_metrics
                    )
                
                return orbit_id, orbit_model
                
        except Exception as e:
            print(f"轨道 {orbit_id} 训练失败: {str(e)}")
            raise e
            
        return None
        
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
            current_accuracy = metrics.get('accuracy', 0.0)
            current_loss = metrics.get('loss', float('inf'))
            
            # 如果有历史数据就使用，没有就只用当前性能
            if self.visualizer:
                history = self.visualizer.satellite_history.get(client.satellite_id, {})
                prev_accuracy = history.get('accuracy', [0.0])[-1] if history.get('accuracy') else 0.0
                weight = (1.0 / (current_loss + 1e-10)) * (1.0 + 0.7 * current_accuracy + 0.3 * prev_accuracy)
            else:
                weight = (1.0 / (current_loss + 1e-10)) * (1.0 + current_accuracy)
            
            weights_results.append((params, weight))
            
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
        
    def aggregate_orbit_neighbors(self, model1: Parameters, model2: Parameters) -> Parameters:
        """融合相邻轨道的模型"""
        params1 = parameters_to_ndarrays(model1)
        params2 = parameters_to_ndarrays(model2)
        
        # 简单平均两个模型的参数
        aggregated_params = []
        for p1, p2 in zip(params1, params2):
            aggregated_params.append((p1 + p2) / 2)
            
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

    def aggregate_metrics(self, results: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """聚合评估指标
        
        Args:
            results: 评估结果列表，每个元素是(样本数, 指标字典)的元组
            
        Returns:
            聚合后的指标字典
        """
        if not results:
            return {
                'accuracy': 0.0,
                'loss': float('inf')
            }
            
        total_examples = sum(num_examples for num_examples, _ in results)
        weighted_accuracy = 0.0
        weighted_loss = 0.0
        
        for num_examples, metrics in results:
            weight = num_examples / total_examples
            weighted_accuracy += metrics.get('accuracy', 0.0) * weight
            weighted_loss += metrics.get('loss', float('inf')) * weight
            
        return {
            'accuracy': weighted_accuracy,
            'loss': weighted_loss
        }

class SatelliteFlowerServer:
    """卫星联邦学习服务器"""
    def __init__(
        self,
        strategy: SatelliteFedAvg,
        orbit_calculator: OrbitCalculator,
        ground_stations: List[GroundStationConfig],
        num_rounds: int = 5,
        min_fit_clients: int = 3,
        min_eval_clients: int = 3,
        min_available_clients: int = 3,
        visualizer: Optional['FederatedLearningVisualizer'] = None
    ):
        """初始化服务器
        
        Args:
            strategy: 联邦学习策略
            orbit_calculator: 轨道计算器
            ground_stations: 地面站列表
            num_rounds: 训练轮数
            min_fit_clients: 最小训练客户端数
            min_eval_clients: 最小评估客户端数
            min_available_clients: 最小可用客户端数
            visualizer: 可视化器（可选）
        """
        self.strategy = strategy
        self.orbit_calculator = orbit_calculator
        self.ground_stations = ground_stations
        self.num_rounds = num_rounds
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.visualizer = visualizer
        
        self.current_round = 0
        self.model = None
        
        # 添加全局指标追踪
        self.global_metrics = {
            'accuracy': [],
            'loss': []
        }
        
        # 初始化卫星列表
        self.satellites = []
        
    async def train_round(self, clients: List[SatelliteFlowerClient]) -> Dict:
        """执行一轮分层训练"""
        try:
            # 更新卫星列表
            self.satellites = [client.config for client in clients]
            
            self.current_round += 1
            print(f"\n{'='*20} 轮次 {self.current_round}/{self.num_rounds} {'='*20}")
            
            # 更新策略的轮次
            self.strategy.current_round = self.current_round
            
            # 1. 轨道内训练
            orbit_tasks = []
            orbit_models = {}
            orbit_metrics = {}
            
            # 按轨道分组
            orbit_groups = {}
            for client in clients:
                orbit_id = client.satellite_id // 11
                if orbit_id not in orbit_groups:
                    orbit_groups[orbit_id] = []
                orbit_groups[orbit_id].append(client)
            
            # 并行执行所有轨道的训练
            for orbit_id, orbit_clients in orbit_groups.items():
                task = self.strategy.train_orbit(orbit_id, orbit_clients)
                orbit_tasks.append(task)
            
            orbit_results = await asyncio.gather(*orbit_tasks)
            
            # 收集轨道模型和指标
            for result in orbit_results:
                if result:
                    orbit_id, model = result
                    orbit_models[orbit_id] = model
            
            # 2. 地面站级聚合
            station_models = []
            current_time = datetime.now()
            
            print("\n开始地面站级聚合...")
            
            # 创建 GroundStation 对象
            ground_station_instances = [
                GroundStation(config, self.orbit_calculator) 
                for config in self.ground_stations
            ]
            
            # 地面站聚合也可以并行
            async def aggregate_station(station: GroundStation):
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
                        print(f"地面站 {station.config.station_id} 完成轨道间聚合，"
                              f"聚合了 {len(visible_orbits)} 个轨道的模型")
                        return station.config.station_id, station_model
                return None
            
            # 并行执行所有地面站的聚合
            station_tasks = [
                aggregate_station(station) 
                for station in ground_station_instances
            ]
            station_results = await asyncio.gather(*station_tasks)
            
            # 收集地面站模型
            station_models = [result for result in station_results if result is not None]
            
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
            aggregated_metrics = self.strategy.aggregate_metrics(eval_results)
            
            # 更新全局指标
            if self.visualizer:
                print(f"更新全局指标: {aggregated_metrics}")
                # 更新可视化器的全局指标
                self.visualizer.update_global_metrics(
                    round=self.current_round,
                    metrics=aggregated_metrics
                )
                # 保存到服务器的全局指标历史
                self.global_metrics['accuracy'].append(aggregated_metrics['accuracy'])
                self.global_metrics['loss'].append(aggregated_metrics['loss'])
            
            print(f"\n本轮训练结果:")
            print(f"- 准确率: {aggregated_metrics['accuracy']:.4f}")
            print(f"- 损失: {aggregated_metrics['loss']:.4f}")
            print(f"- 参与轨道数: {len(orbit_models)}")
            print(f"- 参与地面站数: {len(station_models)}")
            print(f"第 {self.current_round} 轮完成: accuracy={aggregated_metrics['accuracy']:.4f}, loss={aggregated_metrics['loss']:.4f}")
            
            return aggregated_metrics
            
        except Exception as e:
            print(f"轮次 {self.current_round} 训练失败: {str(e)}")
            raise e
        
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
        
    async def start(self):
        """启动联邦学习服务器"""
        try:
            # 创建卫星客户端
            clients = self.create_clients()
            print(f"创建了 {len(clients)} 颗卫星")
            print(f"创建了 {len(self.ground_stations)} 个地面站\n")
            
            # 执行多轮训练
            for _ in range(self.num_rounds):
                await self.train_round(clients)
                
            # 在训练结束后生成可视化结果
            if self.visualizer:
                self.visualizer.plot_satellite_metrics()
                self.visualizer.plot_orbit_metrics()
                self.visualizer.plot_global_metrics()
                
            print("\n保存实验结果...")
            print("实验结果已保存到 results 目录")
            
        except Exception as e:
            print(f"训练失败: {str(e)}")
            raise e