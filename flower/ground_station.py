from datetime import datetime
import numpy as np
import asyncio
from typing import Dict, List, Optional, TYPE_CHECKING

# 避免循环导入
if TYPE_CHECKING:
    from .config import SatelliteConfig
else:
    SatelliteConfig = 'SatelliteConfig'

from .config import GroundStationConfig
from .orbit_utils import OrbitCalculator

class GroundStation:
    def __init__(self, config: GroundStationConfig, orbit_calculator: OrbitCalculator):
        self.config = config
        self.orbit_calculator = orbit_calculator
        self.semi_major_axis = orbit_calculator.earth_radius
        self.eccentricity = 0.0
        self.inclination = 0.0
        self.raan = 0.0
        self.arg_perigee = 0.0
        self.current_version = 0
        
    def get_position(self, current_time: datetime) -> np.ndarray:
        """获取地面站的位置"""
        lat_rad = np.radians(self.config.latitude)
        lon_rad = np.radians(self.config.longitude)
        
        r = self.orbit_calculator.earth_radius
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        
        return np.array([x, y, z])
        
    async def distribute_model(self, model: Dict, satellites: List['SatelliteConfig']) -> List[str]:
        """并行分发模型到可见的卫星"""
        visible_satellites = self.get_visible_satellites(satellites)
        if not visible_satellites:
            return []
            
        tasks = []
        for sat in visible_satellites:
            task = asyncio.create_task(
                self._distribute_to_satellite(model, sat)
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_sats = []
        for sat, result in zip(visible_satellites, results):
            if isinstance(result, Exception):
                print(f"分发到卫星 {sat.sat_id} 失败: {str(result)}")
            else:
                successful_sats.append(sat.sat_id)
                
        return successful_sats
        
    async def _distribute_to_satellite(self, model: Dict, sat_config: 'SatelliteConfig') -> bool:
        """分发模型到单个卫星"""
        try:
            transfer_time = self._calculate_transfer_time(model, sat_config)
            await asyncio.sleep(transfer_time)
            self.current_version += 1
            return True
        except Exception as e:
            print(f"分发错误: {str(e)}")
            return False
            
    def get_visible_satellites(self, satellites: List[SatelliteConfig]) -> List[SatelliteConfig]:
        """获取当前可见的卫星列表"""
        current_time = datetime.now()
        visible_sats = []
        
        for sat in satellites:
            if self.orbit_calculator.check_satellite_visibility(sat, self.config, current_time):
                visible_sats.append(sat)
                if len(visible_sats) >= self.config.max_satellites:
                    break
        
        return visible_sats 