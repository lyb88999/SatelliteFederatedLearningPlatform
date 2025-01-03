import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os

class FederatedLearningVisualizer:
    def __init__(self, save_dir: str = "results"):
        """初始化可视化器
        
        Args:
            save_dir: 保存图表的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储训练历史
        self.satellite_history = {}  # 卫星级别的历史
        self.orbit_history = {}      # 轨道级别的历史
        self.global_history = {      # 全局级别的历史
            'accuracy': [],
            'loss': []
        }
        
    def update_satellite_metrics(self, round: int, satellite_id: int, metrics: Dict[str, float], is_training: bool = False):
        """更新卫星级别的指标"""
        # 确保卫星ID存在
        if satellite_id not in self.satellite_history:
            self.satellite_history[satellite_id] = {
                'train_accuracy': [],
                'train_loss': [],
                'eval_accuracy': [],
                'eval_loss': [],
                'rounds': []
            }
        
        history = self.satellite_history[satellite_id]
        
        # 确保轮次列表存在且长度正确
        while len(history['rounds']) < round:
            history['rounds'].append(len(history['rounds']) + 1)
            history['train_accuracy'].append(0.0)
            history['train_loss'].append(float('inf'))
            history['eval_accuracy'].append(0.0)
            history['eval_loss'].append(float('inf'))
        
        # 更新指标
        if is_training:
            history['train_accuracy'][round-1] = metrics.get('accuracy', 0.0)
            history['train_loss'][round-1] = metrics.get('loss', float('inf'))
        else:
            history['eval_accuracy'][round-1] = metrics.get('accuracy', 0.0)
            history['eval_loss'][round-1] = metrics.get('loss', float('inf'))
        
    def update_orbit_metrics(self, round_num: int, orbit_id: int, metrics: Dict):
        """更新轨道级别的指标"""
        if orbit_id not in self.orbit_history:
            self.orbit_history[orbit_id] = {
                'accuracy': [],
                'loss': []
            }
            
        # 确保列表长度与轮次匹配
        while len(self.orbit_history[orbit_id]['accuracy']) < round_num - 1:
            self.orbit_history[orbit_id]['accuracy'].append(0.0)
            self.orbit_history[orbit_id]['loss'].append(float('inf'))
        
        self.orbit_history[orbit_id]['accuracy'].append(metrics.get('accuracy', 0))
        self.orbit_history[orbit_id]['loss'].append(metrics.get('loss', float('inf')))
        
    def update_global_metrics(self, round: int, metrics: Dict[str, float]):
        """更新全局指标"""
        if 'accuracy' not in self.global_history:
            self.global_history['accuracy'] = []
        if 'loss' not in self.global_history:
            self.global_history['loss'] = []
        
        # 确保列表长度与轮次匹配
        while len(self.global_history['accuracy']) < round - 1:
            self.global_history['accuracy'].append(0.0)
            self.global_history['loss'].append(float('inf'))
        
        self.global_history['accuracy'].append(metrics['accuracy'])
        self.global_history['loss'].append(metrics['loss'])
        
    def plot_satellite_metrics(self):
        """绘制卫星指标图，按轨道聚合显示"""
        if not self.satellite_history:
            print("警告: 没有卫星历史数据可供绘制")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 按轨道聚合卫星数据
        orbit_metrics = {}
        for sat_id, history in self.satellite_history.items():
            orbit_id = sat_id // 11  # 计算轨道ID
            if orbit_id not in orbit_metrics:
                orbit_metrics[orbit_id] = {
                    'train_accuracy': np.zeros_like(history['train_accuracy']),
                    'train_loss': np.zeros_like(history['train_loss']),
                    'eval_accuracy': np.zeros_like(history['eval_accuracy']),
                    'eval_loss': np.zeros_like(history['eval_loss']),
                    'rounds': list(range(1, len(history['rounds']) + 1)),
                    'count': 0
                }
            
            # 累加指标
            orbit_metrics[orbit_id]['train_accuracy'] += np.array(history['train_accuracy'])
            orbit_metrics[orbit_id]['train_loss'] += np.array(history['train_loss'])
            orbit_metrics[orbit_id]['eval_accuracy'] += np.array(history['eval_accuracy'])
            orbit_metrics[orbit_id]['eval_loss'] += np.array(history['eval_loss'])
            orbit_metrics[orbit_id]['count'] += 1
        
        # 计算平均值
        for orbit_id in orbit_metrics:
            count = orbit_metrics[orbit_id]['count']
            if count > 0:
                orbit_metrics[orbit_id]['train_accuracy'] /= count
                orbit_metrics[orbit_id]['train_loss'] /= count
                orbit_metrics[orbit_id]['eval_accuracy'] /= count
                orbit_metrics[orbit_id]['eval_loss'] /= count
        
        # 使用不同的颜色和标记
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        markers = ['o', 's', '^', 'v', 'D', 'p']
        
        # 绘制准确率
        plt.subplot(2, 1, 1)
        for idx, (orbit_id, metrics) in enumerate(orbit_metrics.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            plt.plot(metrics['rounds'], metrics['eval_accuracy'],
                    color=color, marker=marker,
                    label=f'Orbit {orbit_id} (eval)',
                    linestyle='-', markersize=6)
            
            plt.plot(metrics['rounds'], metrics['train_accuracy'],
                    color=color, marker=marker,
                    label=f'Orbit {orbit_id} (train)',
                    linestyle='--', alpha=0.5, markersize=4)
        
        plt.title('Orbit Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 绘制损失
        plt.subplot(2, 1, 2)
        for idx, (orbit_id, metrics) in enumerate(orbit_metrics.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            plt.plot(metrics['rounds'], metrics['eval_loss'],
                    color=color, marker=marker,
                    label=f'Orbit {orbit_id} (eval)',
                    linestyle='-', markersize=6)
            
            plt.plot(metrics['rounds'], metrics['train_loss'],
                    color=color, marker=marker,
                    label=f'Orbit {orbit_id} (train)',
                    linestyle='--', alpha=0.5, markersize=4)
        
        plt.title('Orbit Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 调整布局以适应图例
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # 保存图表
        save_path = os.path.join(self.save_dir, 'satellite_metrics.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"图表已保存到: {save_path}")
        
    def plot_orbit_metrics(self):
        """绘制所有轨道的性能指标"""
        plt.figure(figsize=(20, 10))  # 增大图表尺寸
        
        # 损失图
        plt.subplot(2, 1, 1)
        for orbit_id, history in self.orbit_history.items():
            if history['loss']:  # 只绘制有数据的轨道
                plt.plot(history['loss'], label=f'Orbit {orbit_id}')
        plt.title('Orbit-level Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        if self.orbit_history:  # 只在有数据时添加图例
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # 准确率图
        plt.subplot(2, 1, 2)
        for orbit_id, history in self.orbit_history.items():
            if history['accuracy']:  # 只绘制有数据的轨道
                plt.plot(history['accuracy'], label=f'Orbit {orbit_id}')
        plt.title('Orbit-level Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        if self.orbit_history:  # 只在有数据时添加图例
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.subplots_adjust(right=0.85)  # 调整右边距，为图例留出空间
        plt.savefig(f'{self.save_dir}/orbit_metrics.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_global_metrics(self):
        """绘制全局性能指标"""
        if not self.global_history['accuracy'] or not self.global_history['loss']:
            print("警告: 没有全局历史数据可供绘制")
            return
        
        plt.figure(figsize=(15, 10))
        
        # 准确率图
        plt.subplot(2, 1, 1)
        plt.plot(range(1, len(self.global_history['accuracy']) + 1),
                 self.global_history['accuracy'],
                 'b-', marker='o', label='Global Accuracy')
        plt.title('Global Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # 损失图
        plt.subplot(2, 1, 2)
        plt.plot(range(1, len(self.global_history['loss']) + 1),
                 self.global_history['loss'],
                 'r-', marker='o', label='Global Loss')
        plt.title('Global Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, 'global_metrics.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"全局指标图表已保存到: {save_path}") 