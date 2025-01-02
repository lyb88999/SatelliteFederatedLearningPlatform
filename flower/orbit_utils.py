import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
from .config import SatelliteConfig, GroundStationConfig

class OrbitCalculator:
    def __init__(self, satellite_config: SatelliteConfig, debug_mode: bool = True):
        self.config = satellite_config
        self.earth_radius = 6371  # 地球半径（公里）
        self.orbit_radius = self.earth_radius + self.config.orbit_altitude
        self.debug_mode = debug_mode
        self.max_satellite_communication_range = 1000  # 卫星间最大通信距离(km)
        self.resource_threshold = 0.2  # 资源阈值(20%)
        self.priority_levels = 3  # 优先级等级数
        
    def _calculate_satellite_position(self, time: datetime) -> Tuple[float, float, float]:
        """计算卫星在给定时间的位置"""
        # 计算时间参数
        t = time.timestamp()
        t0 = datetime(time.year, time.month, time.day).timestamp()
        dt = t - t0
        
        # 计算轨道角度
        n = 2 * np.pi / (self.config.orbital_period * 60)  # 平均运动（弧度/秒）
        orbit_angle = n * dt  # 轨道角度
        
        # 1. 计算卫星在轨道平面内的位置（相对于地心）
        x = self.orbit_radius * np.cos(orbit_angle)
        y = self.orbit_radius * np.sin(orbit_angle)
        z = 0.0
        
        # 2. 应用轨道倾角（绕X轴旋转）
        inclination = np.radians(self.config.orbit_inclination)
        y_inclined = y * np.cos(inclination)
        z_inclined = y * np.sin(inclination)
        
        # 3. 应用升交点赤经（绕Z轴旋转）
        raan = np.radians(self.config.ascending_node)
        x_final = x * np.cos(raan) - y_inclined * np.sin(raan)
        y_final = x * np.sin(raan) + y_inclined * np.cos(raan)
        z_final = z_inclined
        
        return (x_final, y_final, z_final)
    
    def _calculate_ground_station_position(self, station: GroundStationConfig, time: datetime) -> Tuple[float, float, float]:
        """计算地面站的位置"""
        lat = np.radians(station.latitude)
        lon = np.radians(station.longitude)
        
        # 计算地方恒星时（考虑地球自转）
        t = time.timestamp()
        t0 = datetime(time.year, time.month, time.day).timestamp()
        dt = t - t0
        sid_time = 2 * np.pi * (dt % 86400) / 86400
        lon_adjusted = lon + sid_time
        
        # 计算地面站的笛卡尔坐标
        x = self.earth_radius * np.cos(lat) * np.cos(lon_adjusted)
        y = self.earth_radius * np.cos(lat) * np.sin(lon_adjusted)
        z = self.earth_radius * np.sin(lat)
        
        return (x, y, z)
    
    def calculate_visibility(self, ground_station: GroundStationConfig, time: datetime) -> bool:
        """计算卫星是否可见"""
        if self.debug_mode:
            # 在调试模式下，每30秒都有15秒的通信窗口
            seconds = time.timestamp()
            return (int(seconds) % 30) < 15
            
        # 正常模式下使用真实的轨道计算
        return self._calculate_real_visibility(ground_station, time)
        
    def _calculate_real_visibility(self, ground_station: GroundStationConfig, time: datetime) -> bool:
        """计算真实的可见性"""
        # 原来的可见性计算代码
        elevation, slant_range = self._get_pass_metrics(ground_station, time)
        return (elevation >= ground_station.min_elevation and 
                slant_range <= ground_station.coverage_radius)
    
    def get_next_window(self, ground_station: GroundStationConfig, 
                       current_time: datetime) -> Tuple[datetime, timedelta]:
        """计算下一个通信窗口的开始时间和持续时间"""
        if self.debug_mode:
            # 在调试模式下，窗口周期为30秒
            seconds = current_time.timestamp()
            current_mod = int(seconds) % 30
            if current_mod < 15:
                # 已经在窗口内
                return current_time, timedelta(seconds=15-current_mod)
            else:
                # 等待下一个窗口
                wait_seconds = 30 - current_mod
                next_window = current_time + timedelta(seconds=wait_seconds)
                return next_window, timedelta(seconds=15)
        
        # 正常模式下使用真实的轨道计算
        return self._calculate_real_next_window(ground_station, current_time)
        
    def _calculate_real_next_window(self, ground_station: GroundStationConfig, 
                                  current_time: datetime) -> Tuple[datetime, timedelta]:
        """计算真实的下一个通信窗口"""
        # 原来的窗口预测代码
        window_start = None
        window_duration = timedelta(minutes=10)
        
        # 向前搜索24小时内的下一个窗口
        for minutes in range(24 * 60):
            test_time = current_time + timedelta(minutes=minutes)
            if self._calculate_real_visibility(ground_station, test_time):
                window_start = test_time
                break
                
        return window_start, window_duration 
    
    def calculate_satellite_visibility(self, sat1_config: SatelliteConfig, 
                                    sat2_config: SatelliteConfig,
                                    time: datetime) -> bool:
        """计算两颗卫星之间是否可见"""
        # 1. 检查是否在同一轨道
        if sat1_config.orbit_id != sat2_config.orbit_id:
            return False
            
        # 2. 计算两颗卫星的位置
        sat1_pos = self._calculate_satellite_position(time)
        
        # 使用相同的轨道参数,但调整平均近点角来计算第二颗卫星的位置
        temp_config = SatelliteConfig(
            orbit_altitude=sat2_config.orbit_altitude,
            orbit_inclination=sat2_config.orbit_inclination,
            orbital_period=sat2_config.orbital_period,
            ground_stations=sat2_config.ground_stations,
            ascending_node=sat2_config.ascending_node,
            mean_anomaly=sat2_config.mean_anomaly,
            orbit_id=sat2_config.orbit_id,
            is_coordinator=sat2_config.is_coordinator
        )
        self.config = temp_config
        sat2_pos = self._calculate_satellite_position(time)
        
        # 3. 计算卫星间距离
        distance = np.sqrt(
            (sat1_pos[0] - sat2_pos[0])**2 +
            (sat1_pos[1] - sat2_pos[1])**2 +
            (sat1_pos[2] - sat2_pos[2])**2
        )
        
        # 4. 检查是否在通信范围内
        if distance > self.max_satellite_communication_range:
            return False
            
        # 5. 检查地球遮挡
        # 计算两颗卫星连线的中点
        midpoint = np.array([
            (sat1_pos[0] + sat2_pos[0])/2,
            (sat1_pos[1] + sat2_pos[1])/2,
            (sat1_pos[2] + sat2_pos[2])/2
        ])
        
        # 计算中点到地心的距离
        midpoint_distance = np.sqrt(np.sum(midpoint**2))
        
        # 如果中点在地球内部,说明被地球遮挡
        if midpoint_distance < self.earth_radius:
            return False
            
        return True
        
    def get_orbit_satellites_in_range(self, sat_config: SatelliteConfig, 
                                    all_satellites: List[SatelliteConfig],
                                    time: datetime) -> List[SatelliteConfig]:
        """获取同一轨道上可通信的卫星列表"""
        visible_satellites = []
        
        for other_sat in all_satellites:
            # 跳过自己
            if other_sat == sat_config:
                continue
                
            # 检查是否在同一轨道且可见
            if (other_sat.orbit_id == sat_config.orbit_id and
                self.calculate_satellite_visibility(sat_config, other_sat, time)):
                visible_satellites.append(other_sat)
                
        return visible_satellites 
    
    def predict_satellite_visibility(self, sat1_config: SatelliteConfig,
                                  sat2_config: SatelliteConfig,
                                  start_time: datetime,
                                  duration_hours: int = 24,
                                  step_minutes: int = 1) -> List[Tuple[datetime, timedelta]]:
        """预测未来时间段内两颗卫星的可见性窗口
        
        Args:
            sat1_config: 第一颗卫星配置
            sat2_config: 第二颗卫星配置
            start_time: 开始预测的时间
            duration_hours: 预测时长(小时)
            step_minutes: 预测步长(分钟)
            
        Returns:
            List[Tuple[datetime, timedelta]]: 可见性窗口列表,每个元素为(开始时间,持续时间)
        """
        windows = []
        current_window_start = None
        
        # 按步长遍历时间段
        for minutes in range(0, duration_hours * 60, step_minutes):
            check_time = start_time + timedelta(minutes=minutes)
            is_visible = self.calculate_satellite_visibility(
                sat1_config,
                sat2_config,
                check_time
            )
            
            # 开始新窗口
            if is_visible and current_window_start is None:
                current_window_start = check_time
                
            # 结束当前窗口
            elif not is_visible and current_window_start is not None:
                window_duration = check_time - current_window_start
                if window_duration.total_seconds() >= 60:  # 只记录持续时间超过1分钟的窗口
                    windows.append((current_window_start, window_duration))
                current_window_start = None
        
        # 检查最后一个窗口
        if current_window_start is not None:
            window_duration = start_time + timedelta(hours=duration_hours) - current_window_start
            if window_duration.total_seconds() >= 60:
                windows.append((current_window_start, window_duration))
        
        return windows
        
    def predict_orbit_communication_schedule(self, 
                                          satellites: List[SatelliteConfig],
                                          start_time: datetime,
                                          duration_hours: int = 24) -> Dict[str, List[Tuple[datetime, List[str]]]]:
        """预测轨道内所有卫星的通信调度计划
        
        Args:
            satellites: 轨道内的所有卫星配置
            start_time: 开始预测的时间
            duration_hours: 预测时长(小时)
            
        Returns:
            Dict[str, List[Tuple[datetime, List[str]]]]: 每个卫星的通信计划
            key: 卫星ID
            value: [(时间点, 可通信的卫星ID列表), ...]
        """
        schedule = {sat.orbit_id: [] for sat in satellites}
        
        # 按10分钟为间隔预测
        for minutes in range(0, duration_hours * 60, 10):
            check_time = start_time + timedelta(minutes=minutes)
            
            # 检查每个卫星的可通信对象
            for sat in satellites:
                visible_sats = self.get_orbit_satellites_in_range(
                    sat,
                    satellites,
                    check_time
                )
                
                if visible_sats:
                    schedule[sat.orbit_id].append((
                        check_time,
                        [other_sat.orbit_id for other_sat in visible_sats]
                    ))
        
        return schedule
        
    def find_best_coordinator(self, 
                            satellites: List[SatelliteConfig],
                            start_time: datetime,
                            duration_hours: int = 24) -> Dict[str, Tuple[str, float]]:
        """选择最佳的协调者节点"""
        scores = {}
        
        # 按轨道分组计算得分
        for orbit_id in set(sat.orbit_id for sat in satellites):
            orbit_satellites = [sat for sat in satellites if sat.orbit_id == orbit_id]
            orbit_scores = {}
            
            for i, sat in enumerate(orbit_satellites):
                sat_id = f"orbit_{orbit_id}_sat_{i}"
                score = 0.0
                
                # 1. 计算与同轨道其他卫星的可见性得分
                for j, other_sat in enumerate(orbit_satellites):
                    if i != j:
                        visible_count = 0
                        for hour in range(duration_hours):
                            check_time = start_time + timedelta(hours=hour)
                            if self.calculate_satellite_visibility(sat, other_sat, check_time):
                                visible_count += 1
                        # 可见性得分：每小时可见性的百分比
                        score += (visible_count / duration_hours) * 100
                
                # 2. 计算与地面站的可见性得分
                station_visible_count = 0
                total_checks = duration_hours * 6  # 每小时检查6次
                for station in sat.ground_stations:
                    for hour in range(duration_hours):
                        for minute in [0, 10, 20, 30, 40, 50]:  # 每小时检查6次
                            check_time = start_time + timedelta(hours=hour, minutes=minute)
                            if self.calculate_visibility(station, check_time):
                                station_visible_count += 1
                
                # 地面站可见性得分：可见时间的百分比
                station_score = (station_visible_count / total_checks) * 100
                score += station_score
                
                # 3. 资源状态得分（模拟值）
                battery_score = np.random.uniform(80, 100)  # 假设电池电量在80-100%之间
                memory_score = 100 - np.random.uniform(20, 40)  # 假设内存使用率在20-40%之间
                cpu_score = 100 - np.random.uniform(10, 30)  # 假设CPU使用率在10-30%之间
                
                resource_score = (battery_score + memory_score + cpu_score) / 3
                score += resource_score
                
                # 4. 如果是当前配置的协调者，给予额外加分
                if sat.is_coordinator:
                    score *= 1.2
                
                # 5. 轨道位置加分（越靠近赤道加分越多）
                equator_proximity = 1 - abs(np.sin(np.radians(sat.orbit_inclination)))
                score += equator_proximity * 50
                
                orbit_scores[sat_id] = score
            
            # 选择轨道内得分最高的卫星
            best_sat_id = max(orbit_scores.items(), key=lambda x: x[1])[0]
            best_score = orbit_scores[best_sat_id]
            scores[str(orbit_id)] = (best_sat_id, best_score)
        
        return scores
        
    def calculate_resource_score(self, sat_config: SatelliteConfig, 
                               battery_level: float,
                               memory_usage: float,
                               cpu_usage: float) -> float:
        """计算卫星资源状态得分"""
        # 归一化处理
        battery_score = battery_level / 100.0
        memory_score = 1.0 - memory_usage / 100.0
        cpu_score = 1.0 - cpu_usage / 100.0
        
        # 加权平均
        resource_score = (
            0.4 * battery_score +  # 电量权重最高
            0.3 * memory_score +
            0.3 * cpu_score
        )
        
        return resource_score
        
    def adaptive_schedule(self,
                         satellites: List[SatelliteConfig],
                         start_time: datetime,
                         resource_states: Dict[str, Dict],
                         task_priorities: Dict[str, int],
                         duration_hours: int = 24) -> Dict[str, List[Dict]]:
        """生成自适应通信调度计划"""
        # 按轨道组织卫星
        satellites_by_orbit = {}
        for i, sat in enumerate(satellites):
            if sat.orbit_id not in satellites_by_orbit:
                satellites_by_orbit[sat.orbit_id] = []
            satellites_by_orbit[sat.orbit_id].append((i, sat))
        
        # 初始化调度计划
        schedule = {
            f"orbit_{sat.orbit_id}_sat_{i}": [] 
            for i, sat in enumerate(satellites)
        }
        
        # 为每个轨道生成调度计划
        for orbit_id, orbit_sats in satellites_by_orbit.items():
            # 计算轨道内卫星的得分
            sat_scores = {}
            for i, sat in orbit_sats:
                sat_id = f"orbit_{orbit_id}_sat_{i}"
                
                # 资源状态得分
                if sat_id in resource_states:
                    res_state = resource_states[sat_id]
                    resource_score = self.calculate_resource_score(
                        sat,
                        res_state.get('battery', 100),
                        res_state.get('memory', 0),
                        res_state.get('cpu', 0)
                    )
                else:
                    resource_score = 1.0
                
                # 优先级得分
                priority = task_priorities.get(sat_id, 1)
                priority_score = priority / self.priority_levels
                
                # 综合得分
                sat_scores[sat_id] = 0.6 * resource_score + 0.4 * priority_score
            
            # 生成轨道内通信计划
            for minutes in range(0, duration_hours * 60, 10):
                check_time = start_time + timedelta(minutes=minutes)
                
                # 找出当前可通信的卫星对
                visible_pairs = []
                for i1, sat1 in orbit_sats:
                    for i2, sat2 in orbit_sats:
                        if i1 != i2 and self.calculate_satellite_visibility(sat1, sat2, check_time):
                            sat1_id = f"orbit_{orbit_id}_sat_{i1}"
                            sat2_id = f"orbit_{orbit_id}_sat_{i2}"
                            pair_score = sat_scores[sat1_id] + sat_scores[sat2_id]
                            visible_pairs.append((sat1_id, sat2_id, pair_score))
                
                # 按得分排序并分配通信
                visible_pairs.sort(key=lambda x: x[2], reverse=True)
                allocated = set()
                
                for sat1_id, sat2_id, _ in visible_pairs:
                    if sat1_id not in allocated and sat2_id not in allocated:
                        # 根据资源状态和优先级确定通信时长
                        duration = timedelta(minutes=max(5, min(15, 
                            int(10 * (sat_scores[sat1_id] + sat_scores[sat2_id]) / 2))))
                        
                        # 添加通信事件
                        schedule[sat1_id].append({
                            'time': check_time,
                            'action': 'send',
                            'target': sat2_id,
                            'priority': task_priorities.get(sat1_id, 1),
                            'duration': duration
                        })
                        
                        schedule[sat2_id].append({
                            'time': check_time,
                            'action': 'receive',
                            'target': sat1_id,
                            'priority': task_priorities.get(sat2_id, 1),
                            'duration': duration
                        })
                        
                        allocated.add(sat1_id)
                        allocated.add(sat2_id)
        
        return schedule
        
    def update_schedule(self,
                       current_schedule: Dict[str, List[Dict]],
                       resource_states: Dict[str, Dict],
                       task_priorities: Dict[str, int]) -> Dict[str, List[Dict]]:
        """动态更新调度计划"""
        # 获取当前时间后的调度
        current_time = datetime.now()
        updated_schedule = {}
        
        for sat_id, events in current_schedule.items():
            # 过滤掉已经过期的事件
            future_events = [
                event for event in events 
                if event['time'] > current_time
            ]
            
            # 检查资源状态
            if sat_id in resource_states:
                res_state = resource_states[sat_id]
                resource_score = self.calculate_resource_score(
                    sat_id,
                    res_state.get('battery', 100),
                    res_state.get('memory', 0),
                    res_state.get('cpu', 0)
                )
                
                # 如果资源不足,取消低优先级任务
                if resource_score < self.resource_threshold:
                    future_events = [
                        event for event in future_events
                        if event['priority'] > 1  # 保留高优先级任务
                    ]
            
            updated_schedule[sat_id] = future_events
            
        return updated_schedule 
    
    def check_communication_window(self, sat1_config: SatelliteConfig, sat2_config: SatelliteConfig) -> bool:
        """检查两颗卫星是否在通信窗口内"""
        if self.debug_mode:
            # 在调试模式下，同一轨道内的卫星总是可以通信
            return sat1_config.orbit_id == sat2_config.orbit_id
            
        # 在实际模式下，需要检查：
        # 1. 卫星间距离
        # 2. 通信窗口时间
        # 3. 其他约束条件
        distance = self.calculate_satellite_distance(sat1_config, sat2_config)
        return distance <= 1000  # 假设最大通信距离为1000km
        
    def calculate_satellite_distance(self, sat1_config: SatelliteConfig, sat2_config: SatelliteConfig) -> float:
        """计算两颗卫星之间的距离（公里）"""
        if self.debug_mode:
            # 在调试模式下，同一轨道内的卫星距离设为100km，不同轨道设为2000km
            if sat1_config.orbit_id == sat2_config.orbit_id:
                return 100.0
            return 2000.0
            
        # TODO: 实现实际的卫星距离计算
        return 0.0 