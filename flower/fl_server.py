import flwr as fl
from flwr.common import (
    Parameters,
    FitIns, 
    FitRes,
    EvaluateIns,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

class SatelliteStrategy(FedAvg):
    """卫星联邦学习策略"""
    
    def __init__(
        self,
        orbit_calculator,
        ground_stations,
        fraction_fit: float = 0.2,
        fraction_evaluate: float = 0.15,
        min_fit_clients: int = 8,
        min_evaluate_clients: int = 6,
        min_available_clients: int = 15,
        visualizer=None,
        debug_mode: bool = True,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_metrics_aggregation_fn=self.aggregate_evaluate_metrics
        )
        self.orbit_calculator = orbit_calculator
        self.ground_stations = ground_stations
        self.visualizer = visualizer
        self.debug_mode = debug_mode
        self.current_round = 0
        self.best_accuracy = 0.0
        self.patience = 10
        self.patience_counter = 0
        self.min_delta = 0.01
        self.orbit_metrics = {}  # 轨道级别指标
        self.station_metrics = {}  # 地面站级别指标
        self.satellite_metrics = {}  # 卫星级别指标
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """配置训练参数"""
        self.current_round = server_round
        
        # 获取可用客户端
        sample_size = self.min_fit_clients
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=self.min_fit_clients
        )
        
        # 为每个客户端配置训练参数
        fit_configurations = []
        for client in clients:
            # 从客户端ID中提取轨道信息
            node_id_str = str(client.node_id)
            hash_value = hash(node_id_str)
            idx = abs(hash_value) % 66  # 66 是卫星总数
            orbit_id = idx // 11    # 每个轨道11颗卫星
            
            config = {
                "orbit_id": orbit_id,
                "current_round": server_round,
                "local_epochs": 1,
            }
            fit_configurations.append((client, FitIns(parameters, config)))
            
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """聚合训练结果"""
        if not results:
            return None, {}
            
        # 按轨道分组聚合
        orbit_models = {}
        orbit_metrics = {}  # 记录每个轨道的指标
        
        for client, fit_res in results:
            # 从客户端ID中提取轨道信息
            node_id_str = str(client.node_id)
            hash_value = hash(node_id_str)
            idx = abs(hash_value) % 66
            orbit_id = idx // 11
            satellite_id = idx
            
            # 记录卫星级别指标
            self.satellite_metrics[f"sat_{satellite_id}"] = {
                "round": server_round,
                "accuracy": float(fit_res.metrics["accuracy"]),  # 确保是标量
                "loss": float(fit_res.metrics["loss"])
            }
            
            # 更新轨道级别指标
            if orbit_id not in orbit_metrics:
                orbit_metrics[orbit_id] = {
                    "accuracy": [],
                    "loss": [],
                    "num_satellites": 0
                }
            orbit_metrics[orbit_id]["accuracy"].append(float(fit_res.metrics["accuracy"]))  # 确保是标量
            orbit_metrics[orbit_id]["loss"].append(float(fit_res.metrics["loss"]))
            orbit_metrics[orbit_id]["num_satellites"] += 1
            
            # 存储模型参数和样本数
            if orbit_id not in orbit_models:
                orbit_models[orbit_id] = []
            orbit_models[orbit_id].append((fit_res.parameters, fit_res.num_examples))
        
        # 计算轨道平均指标
        for orbit_id, metrics in orbit_metrics.items():
            self.orbit_metrics[f"orbit_{orbit_id}"] = {
                "round": server_round,
                "accuracy": float(np.mean(metrics["accuracy"])),  # 确保是标量
                "loss": float(np.mean(metrics["loss"])),
                "active_satellites": metrics["num_satellites"]
            }
            
        # 地面站级别聚合和指标统计
        for station in self.ground_stations:
            visible_metrics = {
                "accuracy": [],
                "loss": [],
                "num_orbits": 0
            }
            
            for orbit_id, metrics in orbit_metrics.items():
                if self.orbit_calculator.check_visibility(station, orbit_id):
                    visible_metrics["accuracy"].extend(metrics["accuracy"])  # 使用extend而不是append
                    visible_metrics["loss"].extend(metrics["loss"])
                    visible_metrics["num_orbits"] += 1
                    
            if visible_metrics["num_orbits"] > 0:
                self.station_metrics[f"station_{station.station_id}"] = {
                    "round": server_round,
                    "accuracy": float(np.mean(visible_metrics["accuracy"])),  # 确保是标量
                    "loss": float(np.mean(visible_metrics["loss"])),
                    "visible_orbits": visible_metrics["num_orbits"]
                }
        
        # 打印分层统计信息
        if self.debug_mode:
            print(f"\n=== Round {server_round} Statistics ===")
            
            # 打印每个卫星的指标
            print("\nSatellite-level metrics:")
            sats_this_round = {k: v for k, v in self.satellite_metrics.items() 
                               if v["round"] == server_round}
            for sat_id, metrics in sorted(sats_this_round.items()):
                print(f"{sat_id}: Acc={metrics['accuracy']:.4f}, "
                      f"Loss={metrics['loss']:.4f}, "
                      f"Orbit={int(sat_id.split('_')[1]) // 11}")  # 计算轨道ID
                      
            print("\nOrbit-level metrics:")
            for orbit_id, metrics in self.orbit_metrics.items():
                if metrics["round"] == server_round:
                    print(f"{orbit_id}: Acc={metrics['accuracy']:.4f}, "
                          f"Loss={metrics['loss']:.4f}, "
                          f"Active Sats={metrics['active_satellites']}")
                    
            print("\nGround station metrics:")
            for station_id, metrics in self.station_metrics.items():
                if metrics["round"] == server_round:
                    print(f"{station_id}: Acc={metrics['accuracy']:.4f}, "
                          f"Loss={metrics['loss']:.4f}, "
                          f"Visible Orbits={metrics['visible_orbits']}")
                    
            print("\nSatellite metrics summary:")
            sats_this_round = {k: v for k, v in self.satellite_metrics.items() 
                              if v["round"] == server_round}
            print(f"Active satellites: {len(sats_this_round)}")
            print(f"Avg Accuracy: {np.mean([m['accuracy'] for m in sats_this_round.values()]):.4f}")
            print(f"Avg Loss: {np.mean([m['loss'] for m in sats_this_round.values()]):.4f}")
            print("="*50)
        
        # 地面站级聚合
        visible_models = self._aggregate_ground_stations(orbit_models)
        
        # 全局聚合
        if not visible_models:
            return None, {}
            
        # 简单平均所有地面站的模型
        weights = [1.0/len(visible_models)] * len(visible_models)
        aggregated_params = self._aggregate_parameters(visible_models, weights)
        
        return aggregated_params, {}  # 移除了 avg_metrics，因为它未定义

    def _aggregate_ground_stations(self, orbit_models: Dict[int, List[Tuple[Parameters, int]]]) -> List[Parameters]:
        """地面站级聚合
        Args:
            orbit_models: Dict[轨道ID, List[（模型参数，样本数）]]
        """
        visible_models = []
        for station in self.ground_stations:
            visible_orbit_params = []
            visible_orbit_weights = []
            
            for orbit_id, models in orbit_models.items():
                # 使用轨道计算器检查可见性
                if self.orbit_calculator.check_visibility(station, orbit_id):
                    if self.visualizer:
                        print(f"地面站 {station.station_id} 可见轨道 {orbit_id}")
                        
                    # 先聚合这个轨道内的所有模型
                    parameters = [p for p, _ in models]
                    weights = [float(n) for _, n in models]
                    orbit_aggregated = self._aggregate_parameters(parameters, weights)
                    
                    # 添加到可见轨道列表
                    visible_orbit_params.append(orbit_aggregated)
                    visible_orbit_weights.append(sum(weights))  # 使用轨道内总样本数作为权重
            
            if visible_orbit_params:
                # 聚合所有可见轨道的模型
                station_model = self._aggregate_parameters(
                    visible_orbit_params,
                    visible_orbit_weights
                )
                visible_models.append(station_model)
            
        return visible_models

    def _aggregate_parameters(self, parameters_list: List[Parameters], 
                             weights: List[float]) -> Parameters:
        """聚合参数
        Args:
            parameters_list: 要聚合的模型参数列表
            weights: 对应的权重列表
        """
        # 转换为numpy数组
        parameters = [parameters_to_ndarrays(p) for p in parameters_list]
        
        # 确保权重是浮点数并归一化
        weights = [float(w) for w in weights]
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # 加权平均每一层
        aggregated_params = []
        for layer_idx in range(len(parameters[0])):
            layer_updates = [params[layer_idx] for params in parameters]
            weighted_layer = np.zeros_like(layer_updates[0])
            for w, update in zip(weights, layer_updates):
                weighted_layer += update * w
            aggregated_params.append(weighted_layer)
        
        return ndarrays_to_parameters(aggregated_params)

    def aggregate_evaluate_metrics(self, metrics):
        """聚合评估指标"""
        if not metrics:
            return {}
            
        # 计算加权平均
        keys = metrics[0][1].keys()
        aggregated = {}
        total_examples = sum(num_examples for num_examples, _ in metrics)
        
        for key in keys:
            weighted_metric = 0.0
            for num_examples, m in metrics:
                weighted_metric += m[key] * num_examples / total_examples
            aggregated[key] = weighted_metric
            
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """聚合评估结果并检查早停"""
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        if aggregated is not None:
            loss_aggregated, metrics_aggregated = aggregated
            current_accuracy = metrics_aggregated.get("accuracy", 0.0)
            
            # 检查是否有显著改进
            if current_accuracy > (self.best_accuracy + self.min_delta):
                self.best_accuracy = current_accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                print(f"Early stopping at round {server_round} (best accuracy: {self.best_accuracy:.4f})")
                return aggregated
                
        return aggregated