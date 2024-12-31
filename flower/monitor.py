from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from .config import SatelliteConfig, GroundStationConfig
from .scheduler import CommunicationWindow

@dataclass
class LinkQuality:
    snr: float  # 信噪比 (dB)
    bit_error_rate: float
    throughput: float  # Mbps

@dataclass
class SatelliteStatus:
    timestamp: datetime
    position: tuple[float, float, float]  # (x, y, z) in km
    velocity: tuple[float, float, float]  # (vx, vy, vz) in km/s
    battery_level: float  # 0-100%
    temperature: float  # 摄氏度
    memory_usage: float  # 0-100%

class Monitor:
    def __init__(self, satellite_config: SatelliteConfig):
        self.config = satellite_config
        self.current_window: Optional[CommunicationWindow] = None
        self.window_history: List[CommunicationWindow] = []
        self.link_quality_history: Dict[str, List[LinkQuality]] = {}
        self.satellite_status_history: List[SatelliteStatus] = []
        
    def update_satellite_status(self, status: SatelliteStatus):
        """更新卫星状态"""
        self.satellite_status_history.append(status)
        
        # 保持历史记录在合理范围内（例如24小时）
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.satellite_status_history = [
            s for s in self.satellite_status_history 
            if s.timestamp > cutoff_time
        ]
    
    def record_link_quality(self, station_id: str, quality: LinkQuality):
        """记录通信链路质量"""
        if station_id not in self.link_quality_history:
            self.link_quality_history[station_id] = []
        self.link_quality_history[station_id].append(quality)
    
    def start_communication_window(self, window: CommunicationWindow):
        """开始新的通信窗口"""
        self.current_window = window
        print(f"开始与地面站 {window.station_id} 的通信窗口")
        print(f"开始时间: {window.start_time}")
        print(f"预计结束时间: {window.end_time}")
        print(f"最大仰角: {window.max_elevation:.1f}°")
        print(f"最小距离: {window.min_distance:.1f} km")
    
    def end_communication_window(self):
        """结束当前通信窗口"""
        if self.current_window:
            self.window_history.append(self.current_window)
            print(f"结束与地面站 {self.current_window.station_id} 的通信窗口")
            self.current_window = None
    
    def get_link_statistics(self, station_id: str, 
                          start_time: datetime, 
                          end_time: datetime) -> Dict[str, float]:
        """获取特定时间段的链路统计信息"""
        if station_id not in self.link_quality_history:
            return {}
        
        # 筛选时间范围内的记录
        records = [
            q for q in self.link_quality_history[station_id]
            if start_time <= q.timestamp <= end_time
        ]
        
        if not records:
            return {}
        
        # 计算统计信息
        snr_values = [r.snr for r in records]
        ber_values = [r.bit_error_rate for r in records]
        throughput_values = [r.throughput for r in records]
        
        return {
            "avg_snr": np.mean(snr_values),
            "min_snr": np.min(snr_values),
            "max_snr": np.max(snr_values),
            "avg_ber": np.mean(ber_values),
            "avg_throughput": np.mean(throughput_values),
            "total_data": np.sum(throughput_values) * (end_time - start_time).total_seconds()
        }
    
    def get_satellite_health(self) -> Dict[str, float]:
        """获取卫星健康状态"""
        if not self.satellite_status_history:
            return {}
        
        latest = self.satellite_status_history[-1]
        return {
            "battery_level": latest.battery_level,
            "temperature": latest.temperature,
            "memory_usage": latest.memory_usage
        } 