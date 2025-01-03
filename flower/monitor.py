from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
from .config import SatelliteConfig
from .scheduler import CommunicationWindow

@dataclass
class LinkQuality:
    """链路质量"""
    signal_strength: float  # dB
    bit_error_rate: float
    latency: float  # ms

@dataclass
class SatelliteStatus:
    """卫星状态"""
    satellite: SatelliteConfig
    is_active: bool
    last_seen: datetime
    link_quality: Optional[LinkQuality] = None

class Monitor:
    """系统监控器"""
    def __init__(self):
        self.satellite_status: Dict[str, SatelliteStatus] = {}
        
    def update_satellite_status(self, satellite: SatelliteConfig, 
                              is_active: bool, link_quality: Optional[LinkQuality] = None):
        """更新卫星状态"""
        self.satellite_status[satellite.sat_id] = SatelliteStatus(
            satellite=satellite,
            is_active=is_active,
            last_seen=datetime.now(),
            link_quality=link_quality
        )
        
    def get_satellite_status(self, satellite_id: str) -> Optional[SatelliteStatus]:
        """获取卫星状态"""
        return self.satellite_status.get(satellite_id)
        
    def get_active_satellites(self) -> List[SatelliteConfig]:
        """获取活跃卫星列表"""
        return [
            status.satellite 
            for status in self.satellite_status.values() 
            if status.is_active
        ] 