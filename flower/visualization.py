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
        
    def update_satellite_metrics(self, round_num: int, satellite_id: int, metrics: Dict):
        """更新单个卫星的指标"""
        if satellite_id not in self.satellite_history:
            self.satellite_history[satellite_id] = {
                'accuracy': [],
                'loss': []
            }
        
        self.satellite_history[satellite_id]['accuracy'].append(metrics.get('accuracy', 0))
        self.satellite_history[satellite_id]['loss'].append(metrics.get('loss', 0))
        
    def update_orbit_metrics(self, round_num: int, orbit_id: int, metrics: Dict):
        """更新轨道级别的指标"""
        if orbit_id not in self.orbit_history:
            self.orbit_history[orbit_id] = {
                'accuracy': [],
                'loss': []
            }
            
        self.orbit_history[orbit_id]['accuracy'].append(metrics.get('accuracy', 0))
        self.orbit_history[orbit_id]['loss'].append(metrics.get('loss', 0))
        
    def update_global_metrics(self, metrics: Dict):
        """更新全局指标"""
        self.global_history['accuracy'].append(metrics.get('accuracy', 0))
        self.global_history['loss'].append(metrics.get('loss', 0))
        
    def plot_satellite_metrics(self):
        """绘制所有卫星的性能指标"""
        plt.figure(figsize=(15, 10))
        
        # 损失图
        plt.subplot(2, 1, 1)
        for sat_id, history in self.satellite_history.items():
            plt.plot(history['loss'], label=f'Satellite {sat_id}')
        plt.title('Satellite-level Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # 准确率图
        plt.subplot(2, 1, 2)
        for sat_id, history in self.satellite_history.items():
            plt.plot(history['accuracy'], label=f'Satellite {sat_id}')
        plt.title('Satellite-level Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/satellite_metrics.png', bbox_inches='tight')
        plt.close()
        
    def plot_orbit_metrics(self):
        """绘制所有轨道的性能指标"""
        plt.figure(figsize=(15, 10))
        
        # 损失图
        plt.subplot(2, 1, 1)
        for orbit_id, history in self.orbit_history.items():
            plt.plot(history['loss'], label=f'Orbit {orbit_id}')
        plt.title('Orbit-level Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # 准确率图
        plt.subplot(2, 1, 2)
        for orbit_id, history in self.orbit_history.items():
            plt.plot(history['accuracy'], label=f'Orbit {orbit_id}')
        plt.title('Orbit-level Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/orbit_metrics.png', bbox_inches='tight')
        plt.close()
        
    def plot_global_metrics(self):
        """绘制全局性能指标"""
        plt.figure(figsize=(15, 10))
        
        # 损失图
        plt.subplot(2, 1, 1)
        plt.plot(self.global_history['loss'], 'b-', label='Global Loss')
        plt.title('Global Loss over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率图
        plt.subplot(2, 1, 2)
        plt.plot(self.global_history['accuracy'], 'r-', label='Global Accuracy')
        plt.title('Global Accuracy over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/global_metrics.png', bbox_inches='tight')
        plt.close() 