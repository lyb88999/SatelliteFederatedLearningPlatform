from typing import List, Dict, Tuple, Optional
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from .fl_server import SatelliteFedAvg
from .client import SatelliteFlowerClient
from .visualization import FederatedLearningVisualizer
import numpy as np
import asyncio

class GroupedSatelliteFedAvg(SatelliteFedAvg):
    """支持分组训练的联邦平均策略"""
    
    def __init__(self, 
                 visualizer: FederatedLearningVisualizer = None,
                 group_size: int = 3,
                 use_grouping: bool = False):
        """
        Args:
            visualizer: 可视化器
            group_size: 每组卫星的数量
            use_grouping: 是否使用分组训练模式
        """
        super().__init__(visualizer=visualizer)
        self.group_size = group_size
        self.use_grouping = use_grouping
        self.current_round = 0
        
    def select_training_satellites(self, orbit_clients: List[SatelliteFlowerClient]) -> List[SatelliteFlowerClient]:
        """选择参与训练的卫星
        
        如果启用分组模式，则每组选择一个代表；否则所有卫星都参与训练
        """
        if not self.use_grouping:
            return orbit_clients
            
        if not orbit_clients:
            return []
            
        # 按照卫星ID排序，确保相邻的卫星在一组
        sorted_clients = sorted(orbit_clients, key=lambda x: x.satellite_id)
        
        # 每group_size个卫星选择一个代表
        training_clients = []
        for i in range(0, len(sorted_clients), self.group_size):
            group = sorted_clients[i:i + self.group_size]
            # 选择组内第一个卫星作为代表
            training_clients.append(group[0])
            print(f"选择卫星 {group[0].satellite_id} 作为组 {i//self.group_size} 的代表")
            
        return training_clients
        
    def get_group_members(self, orbit_clients: List[SatelliteFlowerClient], 
                         training_client: SatelliteFlowerClient) -> List[SatelliteFlowerClient]:
        """获取代表卫星所在组的所有成员"""
        if not self.use_grouping:
            return [training_client]
            
        sorted_clients = sorted(orbit_clients, key=lambda x: x.satellite_id)
        group_idx = sorted_clients.index(training_client) // self.group_size
        start_idx = group_idx * self.group_size
        end_idx = start_idx + self.group_size
        
        return sorted_clients[start_idx:end_idx] 

    def aggregate_orbit(self, orbit_id: int, results: List[Tuple[SatelliteFlowerClient, Tuple[int, Dict[str, float]]]]) -> Optional[Parameters]:
        """轨道内聚合"""
        if not results:
            return None
            
        if not self.use_grouping:
            return super().aggregate_orbit(orbit_id, results)
            
        # 基于性能的权重计算
        weights_results = []
        for client, (num_examples, metrics) in results:
            # 获取当前模型参数
            params = [val.cpu().numpy() for _, val in client.model.state_dict().items()]
            
            # 计算权重 - 代表卫星的权重要考虑其组内成员数
            current_accuracy = metrics.get('accuracy', 0.0)
            current_loss = metrics.get('loss', float('inf'))
            
            # 组大小作为额外权重因子
            # 从results中获取所有客户端
            all_clients = [c for c, _ in results]
            group_size_factor = min(self.group_size, 
                                  len(self.get_group_members(all_clients, client)))
            
            weight = (1.0 / (current_loss + 1e-10)) * (1.0 + current_accuracy) * group_size_factor
            weights_results.append((params, weight))
            
        # 归一化权重并聚合
        total_weight = sum(w for _, w in weights_results)
        aggregated_params = [np.zeros_like(param) for param in weights_results[0][0]]
        
        for parameters, weight in weights_results:
            weight = weight / total_weight
            for i, param in enumerate(parameters):
                aggregated_params[i] += param * weight
                
        print(f"轨道 {orbit_id} 内聚合完成，参与节点数: {len(weights_results)}")
        return ndarrays_to_parameters(aggregated_params)

    async def train_orbit(self, orbit_id: int, orbit_clients: List[SatelliteFlowerClient]):
        """轨道内训练"""
        print(f"\n开始轨道 {orbit_id} 的训练...")
        
        if self.use_grouping:
            # 选择代表卫星进行训练
            training_clients = self.select_training_satellites(orbit_clients)
            print(f"轨道 {orbit_id} 选择了 {len(training_clients)} 个代表卫星进行训练")
            
            # 训练代表卫星
            train_tasks = [client.train() for client in training_clients]
            train_results = await asyncio.gather(*train_tasks)
            
            # 更新代表卫星的训练指标
            for client, (num_examples, metrics) in zip(training_clients, train_results):
                if self.visualizer:
                    self.visualizer.update_satellite_metrics(
                        self.current_round,
                        client.satellite_id,
                        metrics,
                        is_training=True  # 标记为训练指标
                    )
            
            # 聚合代表卫星的模型
            orbit_model = self.aggregate_orbit(
                orbit_id,
                [(client, result) for client, result in zip(training_clients, train_results)]
            )
            
            if orbit_model:
                # 将聚合后的模型分发给轨道内所有卫星
                for client in orbit_clients:
                    await client.set_model(orbit_model)
                
                # 评估所有卫星
                eval_tasks = [client.evaluate() for client in orbit_clients]
                eval_results = await asyncio.gather(*eval_tasks)
                
                # 更新所有卫星的评估指标
                for client, (num_examples, metrics) in zip(orbit_clients, eval_results):
                    if self.visualizer:
                        self.visualizer.update_satellite_metrics(
                            self.current_round,
                            client.satellite_id,
                            metrics,
                            is_training=False  # 标记为评估指标
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
        else:
            # 使用原有的训练方式
            return await super().train_orbit(orbit_id, orbit_clients)
        
        return None 

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