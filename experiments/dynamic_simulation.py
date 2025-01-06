import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import numpy as np
from flower.orbit_utils import AdvancedOrbitCalculator
from experiments.iridium_simulation import create_ground_stations

class DynamicSimulator:
    """动态时间仿真器"""
    
    def __init__(
        self,
        start_time: datetime,
        duration: timedelta,
        time_step: timedelta,
        debug_mode: bool = True
    ):
        self.start_time = start_time
        self.end_time = start_time + duration
        self.time_step = time_step
        self.current_time = start_time
        self.debug_mode = debug_mode
        
        # 初始化轨道计算器
        self.orbit_calculator = AdvancedOrbitCalculator(debug_mode=debug_mode)
        
        # 统计信息
        self.visibility_stats: Dict[str, List[Tuple[datetime, List[int]]]] = {}
        
    async def run_simulation(self):
        """运行动态仿真"""
        # 创建地面站
        ground_stations = await create_ground_stations(self.orbit_calculator)
        
        # 初始化统计信息
        for station in ground_stations:
            self.visibility_stats[station.station_id] = []
        
        # 时间推进循环
        while self.current_time < self.end_time:
            if self.debug_mode:
                print(f"\n=== 时间: {self.current_time} ===")
                
            # 更新每个地面站的可见性
            for station in ground_stations:
                visible_orbits = []
                for orbit_id in range(6):  # 6个轨道面
                    if self.orbit_calculator.check_visibility(
                        station, 
                        orbit_id,
                        current_time=self.current_time
                    ):
                        visible_orbits.append(orbit_id)
                        
                # 记录统计信息
                self.visibility_stats[station.station_id].append(
                    (self.current_time, visible_orbits)
                )
                
                if self.debug_mode:
                    print(f"地面站 {station.station_id}: "
                          f"可见轨道 {visible_orbits}")
            
            # 时间推进
            self.current_time += self.time_step
            
        # 打印统计信息
        self._print_statistics()
    
    def _print_statistics(self):
        """打印详细统计信息"""
        print("\n=== 仿真统计报告 ===")
        print(f"仿真时长: {self.end_time - self.start_time}")
        print(f"时间步长: {self.time_step}")
        print(f"总采样点数: {len(self.visibility_stats[list(self.visibility_stats.keys())[0]])}")
        
        print("\n1. 地面站覆盖情况:")
        print("-" * 50)
        for station_id, stats in self.visibility_stats.items():
            visible_times = sum(len(orbits) > 0 for _, orbits in stats)
            total_times = len(stats)
            coverage = visible_times / total_times
            avg_orbits = np.mean([len(orbits) for _, orbits in stats])
            max_orbits = max(len(orbits) for _, orbits in stats)
            
            print(f"\n{station_id}:")
            print(f"  - 覆盖率: {coverage:.2%}")
            print(f"  - 平均可见轨道数: {avg_orbits:.2f}")
            print(f"  - 最大同时可见轨道数: {max_orbits}")
            print(f"  - 可见时长: {visible_times * self.time_step.total_seconds() / 3600:.1f}小时")
        
        print("\n2. 轨道利用情况:")
        print("-" * 50)
        orbit_stats = {i: 0 for i in range(6)}
        for stats in self.visibility_stats.values():
            for _, orbits in stats:
                for orbit_id in orbits:
                    orbit_stats[orbit_id] += 1
        
        total_observations = sum(orbit_stats.values())
        for orbit_id, count in orbit_stats.items():
            usage = count / total_observations
            print(f"轨道 {orbit_id}: {usage:.2%} ({count} 次被观测)")

async def test_dynamic_simulation():
    """测试动态仿真"""
    # 仿真参数
    start_time = datetime(2024, 1, 1, 0, 0, 0)  # 2024年1月1日 00:00:00
    duration = timedelta(hours=24)  # 仿真24小时
    time_step = timedelta(minutes=5)  # 5分钟步长
    
    # 创建并运行仿真器
    simulator = DynamicSimulator(
        start_time=start_time,
        duration=duration,
        time_step=time_step,
        debug_mode=True
    )
    
    await simulator.run_simulation()

if __name__ == "__main__":
    asyncio.run(test_dynamic_simulation()) 