from datetime import datetime, timedelta
from typing import Dict, List
from .orbit_utils import OrbitCalculator

class TrainingWindow:
    def __init__(self, start_time: datetime, duration: timedelta, coordinator_id: str, priority: int):
        self.start_time = start_time
        self.duration = duration
        self.coordinator_id = coordinator_id
        self.priority = priority

class TrainingScheduler:
    def __init__(self, orbit_calculator: OrbitCalculator):
        self.orbit_calculator = orbit_calculator
    
    def create_training_schedule(
        self,
        satellites: List,
        coordinators: Dict[int, str],
        resource_states: Dict[str, Dict],
        start_time: datetime,
        duration_hours: int
    ) -> Dict[str, List[TrainingWindow]]:
        """创建训练调度计划"""
        schedule = {}
        window_duration = timedelta(minutes=5)  # 默认5分钟的训练窗口
        
        for sat in satellites:
            if not sat.is_coordinator:
                schedule[f"orbit_{sat.orbit_id}_sat_{sat.sat_id}"] = [
                    TrainingWindow(
                        start_time=start_time + timedelta(minutes=10*i),
                        duration=window_duration,
                        coordinator_id=coordinators[sat.orbit_id],
                        priority=1
                    )
                    for i in range(6)  # 每个卫星6个训练窗口
                ]
        
        return schedule 