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
import time
import random
import torch
from collections import OrderedDict
import numpy as np

class SatelliteFedAvg(FedAvg):
    """自定义联邦平均策略"""
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Optional[Parameters]:
        """聚合客户端训练结果"""
        if not results:
            return None
        
        # 提取参数和权重
        weights_results = []
        for client_proxy, fit_res in results:
            if fit_res.parameters:
                weights_results.append(
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                )
        
        if not weights_results:
            return None
            
        # 计算加权平均
        total_examples = sum(num_examples for _, num_examples in weights_results)
        
        # 初始化聚合参数
        aggregated_params = [
            np.zeros_like(param) for param in weights_results[0][0]
        ]
        
        # 加权平均
        for parameters, num_examples in weights_results:
            weight = num_examples / total_examples
            for i, param in enumerate(parameters):
                aggregated_params[i] += param * weight
        
        print(f"聚合了 {len(weights_results)} 个客户端的结果，总样本数: {total_examples}")
        
        # 返回聚合后的参数
        return ndarrays_to_parameters(aggregated_params)
        
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Optional[float]:
        """聚合评估结果"""
        if not results:
            return None
        
        # 计算加权平均的评估指标
        total_examples = sum(num_examples for _, res in results for num_examples in [res.num_examples])
        weighted_metric = 0.0
        
        for _, res in results:
            weight = res.num_examples / total_examples
            weighted_metric += res.metrics.get("accuracy", 0.0) * weight
            
        return weighted_metric

class SatelliteFlowerServer:
    def __init__(
        self,
        client_manager: fl.server.ClientManager,
        strategy: Optional[fl.server.strategy.Strategy] = None,
        orbit_calculator: OrbitCalculator = None,
        debug_mode: bool = True
    ) -> None:
        self.client_manager = client_manager
        self.strategy = strategy
        self.orbit_calculator = orbit_calculator
        self.debug_mode = debug_mode
        self.global_model = None
        self.fit_metrics_aggregated = []
        
        # 初始化轨道协调者映射
        self.orbit_coordinators = {}
        for client in self.client_manager.clients.values():
            if hasattr(client.client, 'config') and client.client.config.is_coordinator:
                self.orbit_coordinators[client.client.config.orbit_id] = client.cid
        
        # 初始化全局模型
        try:
            # 从任意一个非协调者客户端获取初始参数
            for client in self.client_manager.clients.values():
                if hasattr(client.client, 'config') and not client.client.config.is_coordinator:
                    self.global_model = client.get_parameters()
                    print(f"初始化全局模型，参数大小: {len(self.global_model) if self.global_model else 'None'}")
                    if self.global_model and len(self.global_model) > 2:  # 确保模型参数合理
                        break
            else:
                print("警告: 没有找到合适的客户端来初始化全局模型")
                self.global_model = None
        except Exception as e:
            print(f"初始化全局模型失败: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.global_model = None

    def set_global_model(self, model):
        """设置初始全局模型"""
        self.global_model = model

    def fit(self, num_rounds: int) -> Dict[str, List[float]]:
        """执行联邦学习训练"""
        history = {
            'accuracy': [],
            'loss': []
        }
        
        for round_idx in range(num_rounds):
            print(f"\n==================== 轮次 {round_idx + 1}/{num_rounds} ====================")
            
            # 选择客户端
            clients = self.select_clients()
            if not clients:
                print(f"警告: 轮次 {round_idx + 1} 没有可用的客户端")
                continue
                
            # 获取当前模型参数
            parameters = fl.common.ndarrays_to_parameters([
                val.cpu().numpy() for val in self.global_model.state_dict().values()
            ])
            
            # 训练和评估
            results = []
            failures = []  # 记录失败的客户端
            
            for client in clients:
                try:
                    # 使用 Flower 的参数格式
                    res = client.fit(parameters, config={})
                    if res:
                        parameters_updated, num_examples, metrics = res
                        results.append((fl.common.ndarrays_to_parameters(parameters_updated), num_examples, metrics))
                except Exception as e:
                    print(f"客户端训练失败: {str(e)}")
                    failures.append(client.cid)  # 记录失败的客户端ID
                    continue
            
            # 聚合结果
            if results:
                # 聚合参数
                print(f"开始聚合 {len(results)} 个客户端的结果")
                aggregated_result = self.strategy.aggregate_fit(
                    server_round=round_idx,  # 添加轮次参数
                    results=[(client, FitRes(
                        status=Status(code=Code.OK, message="Success"),  # 添加状态
                        parameters=parameters,
                        num_examples=num_examples,
                        metrics=metrics
                    )) for parameters, num_examples, metrics in results],
                    failures=failures
                )
                
                # 更新全局模型
                if aggregated_result is not None:
                    try:
                        state_dict = self.global_model.state_dict()
                        params_dict = zip(state_dict.keys(), fl.common.parameters_to_ndarrays(aggregated_result))
                        state_dict_new = OrderedDict({
                            k: torch.tensor(v) for k, v in params_dict
                        })
                        self.global_model.load_state_dict(state_dict_new)
                        print(f"成功更新全局模型")
                    except Exception as e:
                        print(f"更新全局模型失败: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                else:
                    print("警告: 聚合结果为空")
                    
                self.fit_metrics_aggregated.append(results)
                
                # 评估
                eval_results = []
                eval_failures = []  # 记录评估失败的客户端
                
                eval_parameters = fl.common.ndarrays_to_parameters([
                    val.cpu().numpy() for val in self.global_model.state_dict().values()
                ])
                
                for client in clients:
                    try:
                        eval_res = client.evaluate(eval_parameters, {})
                        if eval_res:
                            eval_results.append(eval_res)
                    except Exception as e:
                        print(f"客户端评估失败: {str(e)}")
                        eval_failures.append(client.cid)
                        continue
                
                if eval_results:
                    # 计算平均指标
                    total_examples = sum(num_examples for _, num_examples, _ in eval_results)
                    weighted_metrics = {
                        'accuracy': 0.0,
                        'loss': 0.0
                    }
                    
                    for _, num_examples, metrics in eval_results:
                        weight = num_examples / total_examples
                        for metric in weighted_metrics:
                            weighted_metrics[metric] += metrics.get(metric, 0.0) * weight
                    
                    history['accuracy'].append(weighted_metrics['accuracy'])
                    history['loss'].append(weighted_metrics['loss'])
                    
                    print(f"\n轮次 {round_idx + 1} 评估结果:")
                    print(f"- 准确率: {weighted_metrics['accuracy']:.4f}")
                    print(f"- 损失: {weighted_metrics['loss']:.4f}")
                    print(f"- 成功客户端数: {len(eval_results)}")
                    if eval_failures:
                        print(f"- 失败客户端: {eval_failures}")
                else:
                    history['accuracy'].append(0.0)
                    history['loss'].append(float('inf'))
            else:
                print(f"警告: 轮次 {round_idx + 1} 没有有效的训练结果")
                history['accuracy'].append(0.0)
                history['loss'].append(float('inf'))
        
        return history

    def select_clients(self) -> List[fl.client.Client]:
        """选择参与训练的客户端"""
        if self.debug_mode:
            # 在调试模式下，返回所有非协调者客户端
            return [
                client for client in self.client_manager.all().values()
                if not client.config.is_coordinator
            ]
        
        # 获取所有客户端
        all_clients = list(self.client_manager.all().values())
        available_clients = []
        
        # 按轨道组织客户端
        orbit_groups = {}
        for client in all_clients:
            if not hasattr(client, 'config'):
                continue
            
            orbit_id = client.config.orbit_id
            if orbit_id not in orbit_groups:
                orbit_groups[orbit_id] = {
                    'coordinator': None,
                    'clients': []
                }
            
            if client.config.is_coordinator:
                orbit_groups[orbit_id]['coordinator'] = client
            else:
                orbit_groups[orbit_id]['clients'].append(client)
        
        # 对每个轨道，选择可以通信的客户端
        for orbit_id, group in orbit_groups.items():
            coordinator = group['coordinator']
            if not coordinator:
                print(f"警告: 轨道 {orbit_id} 没有协调者")
                continue
            
            # 检查每个客户端是否可以与协调者通信
            for client in group['clients']:
                if self.orbit_calculator.check_communication_window(
                    coordinator.config,
                    client.config
                ):
                    available_clients.append(client)
        
        if not available_clients:
            print("警告: 没有可用的客户端")
            return []
        
        print(f"选择了 {len(available_clients)} 个客户端参与训练")
        return available_clients

    def aggregate_metrics(self, metrics_list: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """聚合评估指标"""
        if not metrics_list:
            return {
                'accuracy': 0.0,
                'loss': float('inf')
            }
        
        total_examples = sum(num_examples for num_examples, _ in metrics_list)
        weighted_accuracy = 0.0
        weighted_loss = 0.0
        
        for num_examples, metrics in metrics_list:
            weight = num_examples / total_examples
            weighted_accuracy += metrics.get('accuracy', 0.0) * weight
            weighted_loss += metrics.get('loss', float('inf')) * weight
        
        return {
            'accuracy': weighted_accuracy,
            'loss': weighted_loss
        }