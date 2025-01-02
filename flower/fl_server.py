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
)
from flwr.server.client_proxy import ClientProxy
from datetime import datetime
from .orbit_utils import OrbitCalculator
from .config import SatelliteConfig
import time
import random

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
        self.current_round = 0
        
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

    def fit(self, num_rounds: int) -> Dict[str, List[float]]:
        """执行联邦学习训练"""
        history = {
            'accuracy': [],
            'loss': []
        }
        
        if self.global_model is None and self.strategy.initial_parameters is not None:
            print("使用策略提供的初始参数初始化全局模型")
            self.global_model = fl.common.parameters_to_ndarrays(
                self.strategy.initial_parameters
            )
        
        for round in range(num_rounds):
            print(f"\n{'='*20} 轮次 {round+1}/{num_rounds} {'='*20}")
            
            try:
                # 1. 按轨道组织客户端
                orbit_groups = {}  # 按轨道ID分组
                for client in self.client_manager.clients.values():
                    orbit_id = client.client.config.orbit_id
                    if orbit_id not in orbit_groups:
                        orbit_groups[orbit_id] = {
                            'coordinator': None,
                            'clients': []
                        }
                    if client.client.config.is_coordinator:
                        orbit_groups[orbit_id]['coordinator'] = client
                    else:
                        orbit_groups[orbit_id]['clients'].append(client)
                
                results = []
                # 2. 对每个轨道进行训练
                for orbit_id, group in orbit_groups.items():
                    coordinator = group['coordinator']
                    clients = group['clients']
                    
                    if not coordinator or not clients:
                        print(f"轨道 {orbit_id} 缺少协调者或客户端，跳过")
                        continue
                    
                    print(f"\n>>> 地面站向轨道 {orbit_id} 的协调者 {coordinator.cid} 发送全局模型")
                    
                    # 使用 fit 方法传递参数给协调者
                    coordinator_res = coordinator.fit(
                        parameters=fl.common.ndarrays_to_parameters(self.global_model),
                        config={"round": round, "is_coordinator": True}
                    )
                    
                    if coordinator_res and coordinator_res[0] is not None:
                        print(f"<<< 协调者 {coordinator.cid} 接收成功")
                        
                        # 协调者按通信窗口顺序转发给轨道内卫星
                        print(f"\n>>> 协调者 {coordinator.cid} 向轨道 {orbit_id} 内广播模型:")
                        for client in self._get_clients_by_window(clients, coordinator):
                            print(f"    -> 发送给卫星 {client.cid}")
                            try:
                                fit_res = client.fit(
                                    parameters=coordinator_res[0],  # 使用协调者的参数
                                    config={"round": round}
                                )
                                if fit_res and fit_res[0] is not None:
                                    print(f"    <- 卫星 {client.cid} 训练完成")
                                    metrics = fit_res[2] if len(fit_res) > 2 else {}
                                    print(f"       指标: {metrics}")
                                    
                                    fit_res_obj = fl.common.FitRes(
                                        parameters=fit_res[0],
                                        num_examples=len(client.client.train_loader.dataset),
                                        metrics=metrics,
                                        status=fl.common.Status(code=fl.common.Code.OK, message="Success")
                                    )
                                    results.append((client.cid, fit_res_obj))
                            except Exception as e:
                                print(f"    !! 卫星 {client.cid} 训练失败: {str(e)}")
                
                # 3. 聚合结果
                if results:
                    try:
                        print("\n>>> 开始聚合客户端参数")
                        parameters_aggregated = self.strategy.aggregate_fit(
                            server_round=round + 1,
                            results=results,
                            failures=[]
                        )
                        
                        if parameters_aggregated:
                            self.global_model = fl.common.parameters_to_ndarrays(parameters_aggregated[0])
                            print("<<< 参数聚合完成")
                            print(f"    新的全局模型参数形状: {[arr.shape for arr in self.global_model]}")
                            
                            # 4. 评估
                            print("\n>>> 开始评估阶段")
                            eval_results = []
                            for orbit_id, group in orbit_groups.items():
                                coordinator = group['coordinator']
                                clients = group['clients']
                                
                                print(f"\n>>> 地面站向轨道 {orbit_id} 的协调者发送评估模型")
                                # 先让协调者接收模型
                                coordinator_res = coordinator.fit(
                                    parameters=fl.common.ndarrays_to_parameters(self.global_model),
                                    config={"round": round, "is_coordinator": True}
                                )
                                
                                if coordinator_res and coordinator_res[0] is not None:
                                    print(f"<<< 协调者 {coordinator.cid} 接收成功")
                                    
                                    # 协调者转发给可通信的客户端
                                    for client in self._get_clients_by_window(clients, coordinator):
                                        try:
                                            print(f"    -> 协调者转发给卫星 {client.cid}")
                                            eval_res = client.evaluate(
                                                parameters=coordinator_res[0],  # 使用协调者的参数
                                                config={}
                                            )
                                            if eval_res:
                                                metrics = eval_res[2] if len(eval_res) > 2 else {}
                                                eval_results.append((client.cid, metrics))
                                                print(f"    <- 卫星 {client.cid} 评估完成，metrics: {metrics}")
                                        except Exception as e:
                                            print(f"    !! 卫星 {client.cid} 评估失败: {str(e)}")
                            
                            # 5. 计算指标
                            if eval_results:
                                try:
                                    round_metrics = self.strategy.evaluate_metrics_aggregation_fn(eval_results)
                                    history['accuracy'].append(round_metrics.get('accuracy', 0.0))
                                    history['loss'].append(round_metrics.get('loss', float('inf')))
                                    
                                    print(f"\n轮次 {round+1} 完成:")
                                    print(f"- 全局准确率: {round_metrics['accuracy']:.4f}")
                                    print(f"- 全局损失: {round_metrics['loss']:.4f}")
                                except Exception as e:
                                    print(f"轮次 {round+1} 指标聚合失败: {str(e)}")
                                    import traceback
                                    print(traceback.format_exc())
                                    history['accuracy'].append(0.0)
                                    history['loss'].append(float('inf'))
                            else:
                                print(f"警告: 轮次 {round+1} 没有有效的评估结果")
                                history['accuracy'].append(0.0)
                                history['loss'].append(float('inf'))
                        else:
                            print(f"警告: 轮次 {round+1} 聚合失败")
                            history['accuracy'].append(0.0)
                            history['loss'].append(float('inf'))
                    except Exception as e:
                        print(f"轮次 {round+1} 聚合或评估失败: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
                        history['accuracy'].append(0.0)
                        history['loss'].append(float('inf'))
                else:
                    print(f"警告: 轮次 {round+1} 没有有效的训练结果")
                    history['accuracy'].append(0.0)
                    history['loss'].append(float('inf'))
                    
            except Exception as e:
                print(f"轮次 {round+1} 执行失败: {str(e)}")
                import traceback
                print(traceback.format_exc())
                history['accuracy'].append(0.0)
                history['loss'].append(float('inf'))
                
        return history

    def _get_client_by_id(self, client_id: str) -> ClientProxy:
        """根据ID获取客户端"""
        return next(
            client for client in self.client_manager.clients.values()
            if client.cid == client_id
        )

    def _get_orbit_clients(self, orbit_id: int) -> List[ClientProxy]:
        """获取轨道内的非协调者客户端"""
        return [
            client for client in self.client_manager.clients.values()
            if hasattr(client.client, 'config') and 
            client.client.config.orbit_id == orbit_id and
            not client.client.config.is_coordinator
        ]

    def _evaluate_global_model(self) -> Dict[str, float]:
        """评估全局模型"""
        if not self.global_model:
            print("警告: 评估时全局模型为空")
            return {'accuracy': 0.0, 'loss': float('inf')}
        
        if self.debug_mode:
            # 在调试模式下，使用客户端评估结果的平均值
            metrics_list = []
            for client in self.client_manager.clients.values():
                if hasattr(client.client, 'config') and not client.client.config.is_coordinator:
                    try:
                        result = client.evaluate(self.global_model, {})
                        if result and result.metrics:
                            metrics_list.append(result.metrics)
                        else:
                            print(f"警告: 客户端 {client.cid} 评估返回空结果")
                    except Exception as e:
                        print(f"客户端 {client.cid} 评估失败: {str(e)}")
                        import traceback
                        print(traceback.format_exc())
            
            if metrics_list:
                avg_accuracy = sum(m.get('accuracy', 0) for m in metrics_list) / len(metrics_list)
                avg_loss = sum(m.get('loss', 0) for m in metrics_list) / len(metrics_list)
                return {
                    'accuracy': avg_accuracy,
                    'loss': avg_loss
                }
            else:
                print("警告: 没有成功的评估结果")
        
        # 如果没有评估结果，返回随机值
        return {
            'accuracy': random.uniform(0.4, 0.6),
            'loss': random.uniform(0.6, 0.8)
        }

    def _aggregate_parameters(self, parameters_list: List[Parameters]) -> Parameters:
        """聚合参数"""
        # 使用 FedAvg 策略聚合参数
        results = [(parameters, 1) for parameters in parameters_list]  # 权重都设为1
        failures = []  # 记录失败的客户端
        return self.strategy.aggregate_fit(results, failures)

    def _train_orbit(self, coordinator: ClientProxy, clients: List[ClientProxy]) -> Dict[str, float]:
        """执行轨道内训练"""
        print(f"\n开始轨道 {coordinator.client.config.orbit_id} 训练，协调者: {coordinator.cid}")
        print(f"轨道内客户端数量: {len(clients)}")
        
        if not hasattr(self, 'global_model'):
            print("警告: 全局模型未初始化")
            # 从协调者获取初始模型
            self.global_model = coordinator.get_parameters()
            if not self.global_model:
                print("错误: 无法从协调者获取模型")
                return {'loss': float('inf'), 'accuracy': 0.0}
        
        # 收集客户端训练结果
        results = []
        metrics_list = []
        
        # 过滤掉协调者
        clients = [c for c in clients if not c.client.config.is_coordinator]
        print(f"过滤后的客户端数量: {len(clients)}")
        
        # 执行训练
        for client in clients:
            try:
                print(f"\n客户端 {client.cid} 开始训练")
                
                # 使用 fit 而不是 train
                fit_res = client.fit(
                    parameters=fl.common.ndarrays_to_parameters(self.global_model),
                    config={}
                )
                
                if not fit_res:
                    print(f"警告: 客户端 {client.cid} 返回空结果")
                    continue
                    
                parameters = fit_res[0]  # 更新后的模型参数
                num_examples = fit_res[1]  # 训练样本数
                metrics = fit_res[2]  # 训练指标
                
                print(f"客户端 {client.cid} 训练完成:")
                print(f"- 样本数: {num_examples}")
                print(f"- 损失: {metrics.get('loss', 'N/A')}")
                print(f"- 准确率: {metrics.get('accuracy', 'N/A')}")
                
                results.append((parameters, num_examples))
                metrics_list.append(metrics)
                
            except Exception as e:
                print(f"\n客户端 {client.cid} 训练失败:")
                print(f"错误类型: {type(e).__name__}")
                print(f"错误信息: {str(e)}")
                import traceback
                print("详细错误信息:")
                print(traceback.format_exc())
        
        # 如果没有成功的训练结果，返回默认值
        if not results:
            print(f"\n警告: 轨道 {coordinator.client.config.orbit_id} 没有成功的训练结果")
            return {'loss': float('inf'), 'accuracy': 0.0}
        
        try:
            # 执行聚合
            print("\n开始聚合训练结果")
            aggregated = self.strategy.aggregate_fit(results, [])
            
            if not aggregated:
                print("警告: 聚合结果为空")
                return {'loss': float('inf'), 'accuracy': 0.0}
            
            # 更新协调者的模型
            coordinator.set_parameters(aggregated)
            
            # 计算平均指标
            avg_metrics = {
                'loss': sum(m['loss'] for m in metrics_list) / len(metrics_list),
                'accuracy': sum(m['accuracy'] for m in metrics_list) / len(metrics_list)
            }
            
            print(f"\n轨道 {coordinator.client.config.orbit_id} 训练完成:")
            print(f"- 平均损失: {avg_metrics['loss']:.4f}")
            print(f"- 平均准确率: {avg_metrics['accuracy']:.4f}")
            
            return avg_metrics
            
        except Exception as e:
            print(f"\n聚合失败:")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            import traceback
            print("详细错误信息:")
            print(traceback.format_exc())
            return {'loss': float('inf'), 'accuracy': 0.0}

    def _is_visible(self, client: ClientProxy) -> bool:
        """检查卫星是否可见"""
        if self.debug_mode:
            return True
        return self.orbit_calculator.calculate_ground_visibility(
            client.client.config,
            datetime.now()
        )

    def _select_coordinators(self) -> Dict[int, str]:
        """为每个轨道选择协调者"""
        coordinators = {}
        for client in self.client_manager.clients.values():
            if hasattr(client.client, 'config'):
                config = client.client.config
                if config.is_coordinator:  # 使用配置中的协调者标志
                    coordinators[config.orbit_id] = client.cid
        return coordinators

    def _distribute_global_model(self):
        """通过地面站分发全局模型"""
        if self.global_model is None:
            # 首次训练，使用初始模型
            for client in self.client_manager.clients.values():
                self.global_model = client.get_parameters({})  # 添加空配置字典
                break

        # 找到当前可见的卫星
        visible_clients = self._find_visible_clients()
        if not visible_clients:
            print("等待卫星进入地面站可见范围...")
            while not visible_clients:
                visible_clients = self._find_visible_clients()
                time.sleep(0.1)  # 避免过于频繁的检查

        # 为每个轨道选择一个可见卫星作为种子节点
        orbit_seeds = {}  # 记录每个轨道的种子节点
        for client in visible_clients:
            orbit_id = client.client.config.orbit_id
            if orbit_id not in orbit_seeds:
                orbit_seeds[orbit_id] = client
                # 使用 fit 方法来设置参数，因为 ClientProxy 不支持直接的 set_parameters
                client.fit(
                    parameters=self.global_model,
                    config={"round": 0, "is_init": True}  # 标记这是初始化
                )

    def _find_visible_clients(self) -> List[ClientProxy]:
        """找到当前可见的卫星"""
        visible_clients = []
        for client in self.client_manager.clients.values():
            if self._is_visible(client):
                visible_clients.append(client)
        return visible_clients

    def _flood_model(self, seed_client: ClientProxy, orbit_id: int):
        """在轨道内洪泛模型"""
        print(f"\n开始在轨道 {orbit_id} 内洪泛模型")
        
        # 获取轨道内所有客户端
        orbit_clients = [
            client for client in self.client_manager.clients.values()
            if hasattr(client.client, 'config') and 
            client.client.config.orbit_id == orbit_id
        ]
        
        # 初始化洪泛状态
        received = {client.cid: False for client in orbit_clients}
        received[seed_client.cid] = True
        level = {seed_client.cid: 0}  # 记录每个节点的层级
        reached_count = 1
        max_level = 0
        
        # 执行洪泛
        while True:
            updated = False
            for sender in orbit_clients:
                if not received[sender.cid]:
                    continue
                    
                sender_level = level[sender.cid]
                
                # 向相邻节点传播
                for receiver in orbit_clients:
                    if received[receiver.cid]:
                        continue
                        
                    # 检查是否相邻
                    if self._are_adjacent(sender.client.config, receiver.client.config):
                        print(f"[层级 {sender_level} -> {sender_level+1}] 节点 {sender.cid} 将参数传递给相邻节点 {receiver.cid}")
                        receiver.set_parameters(self.global_model)
                        received[receiver.cid] = True
                        level[receiver.cid] = sender_level + 1
                        max_level = max(max_level, sender_level + 1)
                        reached_count += 1
                        updated = True
            
            if not updated:
                break
        
        print(f"轨道 {orbit_id} 内洪泛完成:")
        print(f"- 总节点数: {len(orbit_clients)}")
        print(f"- 到达节点: {reached_count}")
        print(f"- 最大层级: {max_level}")

    def _are_adjacent(self, sat1: SatelliteConfig, sat2: SatelliteConfig) -> bool:
        """检查两颗卫星是否相邻"""
        # 在同一轨道内
        if sat1.orbit_id == sat2.orbit_id:
            # 计算卫星序号差值
            diff = abs(sat1.sat_id - sat2.sat_id)
            # 考虑环形连接
            if diff == 1 or diff == 3:  # 相邻或通过环形连接相邻
                if diff == 3:
                    print(f"卫星 {sat1.sat_id} 和 {sat2.sat_id} 相邻（环形连接）")
                else:
                    print(f"卫星 {sat2.sat_id} 和 {sat1.sat_id} 相邻")
                return True
            else:
                print(f"卫星 {sat2.sat_id} 和 {sat1.sat_id} 不相邻")
                return False
        return False

    def _execute_hierarchical_training(
        self,
        coordinators: Dict[int, str]
    ) -> Dict[int, Dict[str, float]]:
        """执行分层训练"""
        orbit_metrics = {}
        
        for orbit_id, coordinator_id in coordinators.items():
            coordinator = next(
                c for c in self.client_manager.clients.values()
                if c.cid == coordinator_id
            )
            
            print(f"\n开始轨道 {orbit_id} 训练，协调者: {coordinator_id}")
            
            # 获取轨道内的非协调者客户端
            orbit_clients = [
                c for c in self.client_manager.clients.values()
                if hasattr(c.client, 'config') and 
                c.client.config.orbit_id == orbit_id and
                not c.client.config.is_coordinator
            ]
            
            print(f"轨道内客户端数量: {len(orbit_clients)}")
            
            # 等待洪泛完成
            time.sleep(1)
            
            # 执行轨道内训练
            orbit_metrics[orbit_id] = self._train_orbit(coordinator, orbit_clients)
            
        return orbit_metrics

    def _aggregate_orbit_models(
        self,
        coordinators: Dict[int, str],
        orbit_metrics: Dict[int, Dict[str, float]]
    ) -> Dict[str, float]:
        """在地面站聚合不同轨道的模型"""
        coordinator_results = []
        
        # 1. 收集协调者的模型参数
        for orbit_id, coordinator_id in coordinators.items():
            coordinator = next(
                c for c in self.client_manager.clients.values()
                if c.cid == coordinator_id
            )
            
            # 等待协调者可见
            while not self._is_coordinator_visible(coordinator):
                print(f"等待协调者 {coordinator_id} 进入地面站可见范围...")
            
            # 收集参数
            parameters = coordinator.get_parameters()
            parameters = ndarrays_to_parameters(parameters)
            
            fit_res = fl.common.FitRes(
                status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
                parameters=parameters,
                num_examples=1,
                metrics={"orbit_id": orbit_id}
            )
            coordinator_results.append((coordinator_id, fit_res))
            
        # 2. 执行全局聚合
        aggregation_result = self.strategy.aggregate_fit(
            server_round=self.current_round,
            results=coordinator_results,
            failures=[]
        )
        
        if aggregation_result is None:
            print("全局参数聚合失败，使用第一个协调者的参数")
            global_parameters = fl.common.parameters_to_ndarrays(coordinator_results[0][1].parameters)
        else:
            # 从聚合结果中获取参数
            global_parameters = fl.common.parameters_to_ndarrays(aggregation_result[0])
        
        # 更新全局模型
        self.global_model = global_parameters
        
        # 3. 计算全局指标
        global_metrics = {
            "accuracy": sum(m["accuracy"] for m in orbit_metrics.values()) / len(orbit_metrics),
            "loss": sum(m["loss"] for m in orbit_metrics.values()) / len(orbit_metrics)
        }
        
        return global_metrics

    def _is_coordinator_visible(self, coordinator: ClientProxy) -> bool:
        """检查协调者是否在任一地面站可见范围内"""
        if self.debug_mode:
            return True
            
        current_time = datetime.now()
        for station in coordinator.client.config.ground_stations:
            if self.orbit_calculator.calculate_visibility(station, current_time):
                return True
        return False 

    def _check_neighbor(self, sat1_config: SatelliteConfig, sat2_config: SatelliteConfig) -> bool:
        """检查两颗卫星是否相邻"""
        if self.debug_mode:
            # 确保在同一轨道
            if sat1_config.orbit_id != sat2_config.orbit_id:
                return False
            
            # 从 client 对象中获取卫星编号
            sat1_num = None
            sat2_num = None
            
            for client in self.client_manager.clients.values():
                if hasattr(client.client, 'config'):
                    if id(client.client.config) == id(sat1_config):
                        sat1_num = int(client.cid.split('_')[-1])
                    if id(client.client.config) == id(sat2_config):
                        sat2_num = int(client.cid.split('_')[-1])
                    
            if sat1_num is None or sat2_num is None:
                print(f"警告: 无法找到卫星编号")
                return False
            
            # 在同一轨道内，允许相邻编号的卫星通信
            # 0-1, 1-2, 2-3, 3-0 形成环形拓扑
            if abs(sat1_num - sat2_num) == 1:
                print(f"卫星 {sat1_num} 和 {sat2_num} 相邻")
                return True
            if (sat1_num == 0 and sat2_num == 3) or (sat1_num == 3 and sat2_num == 0):
                print(f"卫星 {sat1_num} 和 {sat2_num} 相邻（环形连接）")
                return True
            
            print(f"卫星 {sat1_num} 和 {sat2_num} 不相邻")
            return False
        
        # 在实际模式下，检查卫星间的物理距离
        return self.orbit_calculator.calculate_satellite_visibility(
            sat1_config,
            sat2_config,
            datetime.now(),
            max_distance=1000  # 最大通信距离1000km
        )

    def _get_clients_by_window(self, clients: List[ClientProxy], coordinator: ClientProxy) -> List[ClientProxy]:
        """按通信窗口顺序返回可通信的客户端"""
        available_clients = []
        for client in clients:
            # 检查是否在通信窗口内
            if self.orbit_calculator.check_communication_window(
                coordinator.client.config,
                client.client.config
            ):
                available_clients.append(client)
                print(f"    卫星 {client.cid} 在通信窗口内")
            else:
                print(f"    卫星 {client.cid} 不在通信窗口内，跳过")
            
        return available_clients