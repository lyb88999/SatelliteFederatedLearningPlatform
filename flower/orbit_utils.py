import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Union, TYPE_CHECKING

# 避免循环导入
if TYPE_CHECKING:
    from .ground_station import GroundStation
    from .config import SatelliteConfig, GroundStationConfig
else:
    SatelliteConfig = 'SatelliteConfig'
    GroundStation = 'GroundStation'
    GroundStationConfig = 'GroundStationConfig'

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy.coordinates import CartesianRepresentation

# 移除 from poliastro.util import time_range
# 替代方案：自己实现 time_range 函数
def time_range(start_time: datetime, end_time: datetime, step: timedelta) -> List[datetime]:
    """生成时间序列
    
    Args:
        start_time: 开始时间
        end_time: 结束时间
        step: 时间步长
        
    Returns:
        List[datetime]: 时间序列
    """
    times = []
    current = start_time
    while current <= end_time:
        times.append(current)
        current += step
    return times

class OrbitCalculator:
    """轨道计算器"""
    def __init__(self, debug_mode=False, use_simplified=True):
        """初始化轨道计算器
        
        Args:
            debug_mode: 是否启用调试模式
            use_simplified: 是否使用简化的可见性计算
        """
        self.debug_mode = debug_mode
        self.use_simplified = use_simplified
        self.earth_radius = 6371.0  # 地球半径(km)
        self.earth_mu = 398600.4418  # 地球引力常数(km³/s²)
        self.max_communication_distance = 5000.0  # 最大通信距离(km)
        
    def check_satellite_visibility(self, 
                                 satellite: 'SatelliteConfig',
                                 ground_station: Union['GroundStation', 'GroundStationConfig'],
                                 current_time: datetime) -> bool:
        """检查卫星是否可见于地面站"""
        config = ground_station.config if hasattr(ground_station, 'config') else ground_station
        
        # 计算卫星位置和地面站位置
        sat_pos = np.array(self.calculate_satellite_position(satellite, current_time))
        station_pos = self.calculate_ground_station_position(config)
        
        # 计算距离和仰角
        distance = np.linalg.norm(sat_pos - station_pos)
        elevation = self._calculate_elevation(sat_pos, station_pos)
        
        if self.debug_mode:
            print(f"Distance: {distance:.1f} km")
            print(f"Elevation: {elevation:.1f}°")
            print(f"Max range: {config.max_range} km")
            print(f"Min elevation: {config.min_elevation}°")
        
        # 检查可见性条件
        is_visible = (
            distance <= config.max_range and  # 在最大通信距离内
            elevation >= config.min_elevation  # 满足最小仰角要求
        )
        
        return is_visible
        
    def calculate_satellite_position(self, sat_config: SatelliteConfig, time: datetime) -> Tuple[float, float, float]:
        """计算卫星在给定时间的位置"""
        try:
            # 计算从历元开始经过的时间（秒）
            dt = (time - sat_config.epoch).total_seconds()
            
            # 计算轨道周期（秒）
            orbital_period = 2 * np.pi * np.sqrt(sat_config.semi_major_axis**3 / self.earth_mu)
            
            # 计算当前相位角
            mean_motion = 2 * np.pi / orbital_period
            mean_anomaly = mean_motion * dt
            
            # 计算位置
            r = sat_config.semi_major_axis
            inc = np.radians(sat_config.inclination)
            raan = np.radians(sat_config.raan)
            arg_perigee = np.radians(sat_config.arg_perigee)
            
            # 轨道平面内的位置
            x_orbit = r * np.cos(mean_anomaly + arg_perigee)
            y_orbit = r * np.sin(mean_anomaly + arg_perigee)
            
            # 转换到地心惯性坐标系
            x = (x_orbit * np.cos(raan) - y_orbit * np.cos(inc) * np.sin(raan))
            y = (x_orbit * np.sin(raan) + y_orbit * np.cos(inc) * np.cos(raan))
            z = y_orbit * np.sin(inc)
            
            return (float(x), float(y), float(z))
            
        except Exception as e:
            print(f"位置计算错误: {str(e)}")
            return (0.0, 0.0, 0.0)
            
    def calculate_satellite_distance(self, sat1: SatelliteConfig, sat2: SatelliteConfig, 
                                  time: datetime) -> float:
        """计算两颗卫星之间的距离"""
        pos1 = self.calculate_satellite_position(sat1, time)
        pos2 = self.calculate_satellite_position(sat2, time)
        return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(pos1, pos2)))
        
    def calculate_orbital_period(self, semi_major_axis: float) -> float:
        """计算轨道周期（秒）"""
        return 2 * np.pi * np.sqrt(semi_major_axis**3 / self.earth_mu)

    def check_communication_window(self, sat1: SatelliteConfig, sat2: SatelliteConfig) -> bool:
        """检查两颗卫星是否在通信窗口内"""
        return self.check_satellite_visibility(sat1, sat2, datetime.now())

    def _calculate_satellite_position(self, time: datetime) -> np.ndarray:
        """内部方法：计算卫星位置"""
        if self.current_satellite is None:
            # 如果没有设置当前卫星，返回原点
            return np.array([0.0, 0.0, 0.0])
        return np.array(self.calculate_satellite_position(self.current_satellite, time))

    def schedule_adaptive(self, satellites: List[SatelliteConfig], start_time: datetime,
                         resource_states: Dict, task_priorities: Dict, 
                         duration_hours: int) -> Dict:
        """自适应调度"""
        schedule = {}
        for sat in satellites:
            sat_id = f"orbit_{sat.orbit_id}_sat_{sat.sat_id}"
            schedule[sat_id] = []
        return schedule

    def select_best_coordinator(self, satellites: List[SatelliteConfig], 
                              start_time: datetime, duration_hours: int) -> Dict:
        """选择最佳协调者"""
        coordinator_scores = {}
        for sat in satellites:
            orbit_id = sat.orbit_id
            if orbit_id not in coordinator_scores:
                coordinator_scores[orbit_id] = (f"orbit_{orbit_id}_sat_{sat.sat_id}", 1.0)
        return coordinator_scores 

    def calculate_ground_track(self, sat_config: SatelliteConfig, 
                             start_time: datetime, duration_hours: float) -> List[Tuple[float, float]]:
        """计算地面轨迹
        
        Args:
            sat_config: 卫星配置
            start_time: 开始时间
            duration_hours: 持续时间（小时）
            
        Returns:
            List[Tuple[float, float]]: 经纬度列表 [(lon1, lat1), (lon2, lat2), ...]
        """
        if self.debug_mode:
            return [(0.0, 0.0), (10.0, 10.0)]  # 调试模式返回固定值
            
        track_points = []
        time_step = duration_hours * 3600 / 100  # 将时间分成100个点
        
        for i in range(100):
            current_time = start_time + timedelta(seconds=i * time_step)
            pos = self.calculate_satellite_position(sat_config, current_time)
            
            # 将笛卡尔坐标转换为经纬度
            x, y, z = pos
            r = np.sqrt(x*x + y*y + z*z)
            lat = np.arcsin(z/r) * 180/np.pi
            lon = np.arctan2(y, x) * 180/np.pi
            
            track_points.append((lon, lat))
            
        return track_points 

    def calculate_ground_station_position(self, station_config: 'GroundStationConfig') -> np.ndarray:
        """计算地面站的笛卡尔坐标
        
        Args:
            station_config: 地面站配置
            
        Returns:
            np.ndarray: [x, y, z] 笛卡尔坐标
        """
        # 将经纬度转换为弧度
        lat = np.radians(station_config.latitude)
        lon = np.radians(station_config.longitude)
        
        # 计算地面站的笛卡尔坐标
        x = self.earth_radius * np.cos(lat) * np.cos(lon)
        y = self.earth_radius * np.cos(lat) * np.sin(lon)
        z = self.earth_radius * np.sin(lat)
        
        return np.array([x, y, z]) 

    def _calculate_elevation(self, sat_pos: np.ndarray, station_pos: np.ndarray) -> float:
        """计算仰角（度）"""
        # 计算地面站到卫星的向量
        r_vector = sat_pos - station_pos
        
        # 计算地面站的地心方向单位向量
        station_unit = station_pos / np.linalg.norm(station_pos)
        
        # 计算夹角
        cos_angle = np.dot(r_vector, station_unit) / np.linalg.norm(r_vector)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))  # 添加clip避免数值误差
        
        # 转换为仰角
        elevation = 90 - np.degrees(angle)
        return elevation

    def _check_earth_obstruction(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """检查地球是否遮挡了两点之间的视线
        
        Args:
            pos1: 第一个点的位置向量 [x, y, z]
            pos2: 第二个点的位置向量 [x, y, z]
            
        Returns:
            bool: True 如果地球遮挡了视线，False 否则
        """
        # 计算两点之间的向量
        r = pos2 - pos1
        dist = np.linalg.norm(r)
        
        # 计算pos1到地心的距离
        r1 = np.linalg.norm(pos1)
        
        # 如果pos1在地球表面，计算与地平面的夹角
        if abs(r1 - self.earth_radius) < 1.0:  # 允许1km的误差
            # 计算地平面法向量（指向天顶）
            zenith = pos1 / r1
            # 计算视线向量的单位向量
            sight = r / dist
            # 计算夹角的余弦值
            cos_angle = np.dot(zenith, sight)
            # 如果夹角大于90度，说明视线被地球遮挡
            if cos_angle < 0:
                return True
        
        # 计算视线到地心的最短距离
        # 使用向量代数计算点到直线的最短距离
        p = np.dot(pos1, r) / dist
        d = np.sqrt(np.dot(pos1, pos1) - p*p)
        
        # 如果最短距离小于地球半径，说明地球遮挡了视线
        return d < self.earth_radius 

    def check_visibility(self, ground_station, orbit_id):
        """检查地面站是否可以看到指定轨道的卫星"""
        if self.use_simplified:
            return self._simplified_visibility(ground_station, orbit_id)
        else:
            return self._accurate_visibility(ground_station, orbit_id)

    def _simplified_visibility(self, ground_station, orbit_id):
        """使用简化模型检查可见性"""
        station_lat = ground_station.latitude
        orbit_id_int = int(orbit_id)
        
        # 基于纬度的主要可见轨道
        if abs(station_lat) > 60:  # 高纬度
            primary_orbits = {0, 5}
        elif abs(station_lat) > 30:  # 中纬度
            primary_orbits = {1, 2, 3}
        else:  # 低纬度
            primary_orbits = {2, 3, 4}
            
        # 相邻轨道（只添加一个相邻轨道）
        if orbit_id_int in primary_orbits:
            return True
        elif any(abs(orbit_id_int - p) == 1 for p in primary_orbits):
            # 50%的概率看到相邻轨道
            return np.random.random() < 0.5
                
        return False

    def _accurate_visibility(self, ground_station, orbit_id):
        """使用精确模型检查可见性"""
        # 获取地面站位置
        station_pos = self.calculate_ground_station_position(ground_station)
        
        # 获取轨道上的卫星位置
        orbit_satellites = self._get_orbit_satellites(orbit_id)
        
        # 检查是否至少有一颗卫星可见
        for sat_pos in orbit_satellites:
            # 计算仰角
            elevation = self._calculate_elevation(sat_pos, station_pos)
            
            # 检查最小仰角要求（通常为5-10度）
            if elevation < 10:
                continue
                
            # 检查地球遮挡
            if self._check_earth_obstruction(station_pos, sat_pos):
                continue
                
            # 检查通信距离
            distance = np.linalg.norm(sat_pos - station_pos)
            if distance > self.max_communication_distance:
                continue
                
            if self.debug_mode:
                print(f"卫星可见: 仰角={elevation:.1f}°, 距离={distance:.1f}km")
                
            return True
            
        return False

    def _get_orbit_satellites(self, orbit_id):
        """获取轨道上所有卫星的位置"""
        satellites = []
        num_sats = 11  # 每个轨道11颗卫星
        
        for i in range(num_sats):
            # 计算卫星在轨道上的位置
            mean_anomaly = 360.0 * i / num_sats  # 平均角度分布
            pos = self._calculate_satellite_position(orbit_id, mean_anomaly)
            satellites.append(pos)
            
        return satellites

    def _calculate_satellite_position(self, orbit_id, mean_anomaly):
        """计算卫星位置"""
        # 铱星星座参数
        a = self.earth_radius + 780  # 轨道半长轴
        e = 0.0  # 偏心率
        i = np.radians(86.4)  # 轨道倾角
        
        # 计算升交点赤经
        raan = np.radians(360.0 * orbit_id / 6)  # 6个轨道平面
        
        # 将平近点角转换为弧度
        M = np.radians(mean_anomaly)
        
        # 计算卫星位置（这里使用简化的开普勒轨道）
        x = a * (np.cos(M))
        y = a * (np.sin(M) * np.cos(i))
        z = a * (np.sin(M) * np.sin(i))
        
        # 应用升交点赤经旋转
        pos = np.array([
            x * np.cos(raan) - y * np.sin(raan),
            x * np.sin(raan) + y * np.cos(raan),
            z
        ])
        
        return pos 

class AdvancedOrbitCalculator:
    """高级轨道计算器，提供更真实的可见性计算"""
    
    def __init__(self, debug_mode=False):
        self.earth_radius = 6371.0  # 地球半径(km)
        self.min_elevation = 5.0    # 降低最小仰角到5度
        self.max_range = 2500.0     # 最大通信距离
        self.debug_mode = debug_mode
        
        # 轨道相位差（每个轨道面之间的相位差）
        self.phase_difference = 360.0 / (6 * 11)  # 6个轨道面，每个11颗卫星
        
    def check_visibility(self, ground_station, orbit_id, current_time: datetime = None) -> bool:
        """检查卫星是否对地面站可见
        Args:
            ground_station: 地面站对象
            orbit_id: 轨道ID
            current_time: 当前时间点，默认为None表示使用当前系统时间
        """
        gs_pos = self._get_ground_station_position(
            ground_station.latitude,
            ground_station.longitude,
            getattr(ground_station, 'altitude', 0.0)
        )
        
        visible_satellites = []  # 记录可见卫星
        
        # 检查轨道面上的所有卫星
        num_sats = 11  # 每个轨道11颗卫星
        for i in range(num_sats):
            mean_anomaly = i * (360.0 / num_sats)
            sat_pos = self._get_satellite_position(
                orbit_id, 
                mean_anomaly=mean_anomaly,
                time=current_time
            )
            
            # 计算距离
            distance = np.linalg.norm(sat_pos - gs_pos)
            if distance > self.max_range:
                continue
                
            # 计算仰角和可见性
            elevation = self._calculate_elevation(gs_pos, sat_pos)
            if elevation >= self.min_elevation:
                is_visible = self._check_line_of_sight(gs_pos, sat_pos)
                if is_visible:
                    visible_satellites.append((i, elevation, distance))
        
        # 如果有可见卫星，打印信息并返回True
        if visible_satellites and self.debug_mode:
            print(f"\n地面站 {ground_station.station_id} -> 轨道 {orbit_id}:")
            for sat_idx, elev, dist in visible_satellites:
                print(f"  卫星 {sat_idx}: 仰角={elev:.1f}°, 距离={dist:.1f} km")
        
        return len(visible_satellites) > 0
    
    def _get_ground_station_position(self, lat: float, lon: float, alt: float = 0) -> np.ndarray:
        """计算地面站的ECEF坐标"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # WGS84椭球体参数
        a = self.earth_radius
        f = 1/298.257223563  # 扁率
        e2 = 2*f - f*f       # 第一偏心率平方
        
        # 计算卯酉圈曲率半径
        N = a / np.sqrt(1 - e2*np.sin(lat_rad)**2)
        
        # ECEF坐标
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N*(1-e2) + alt) * np.sin(lat_rad)
        
        return np.array([x, y, z])
    
    def _get_satellite_position(self, orbit_id: int, mean_anomaly: float = 0.0, time: datetime = None) -> np.ndarray:
        """计算卫星位置"""
        if time is None:
            time = datetime.now()
            
        # 铱星星座参数
        altitude = 780.0  # 轨道高度(km)
        inclination = np.radians(86.4)  # 轨道倾角
        num_planes = 6
        
        # 计算升交点赤经
        raan = np.radians(orbit_id * 360.0 / num_planes)
        
        # 添加轨道相位差
        phase_offset = orbit_id * self.phase_difference
        mean_anomaly = (mean_anomaly + phase_offset) % 360.0
        
        # 计算轨道进动
        if time:
            epoch = datetime(2024, 1, 1)
            dt = (time - epoch).total_seconds()
            orbital_period = 2 * np.pi * np.sqrt((self.earth_radius + altitude)**3 / 398600.4418)
            nodal_precession = dt * (2.0 * np.pi / (orbital_period * 365.25))
            raan += nodal_precession
        
        # 将平近点角转换为弧度
        M = np.radians(mean_anomaly)
        
        # 计算卫星位置
        r = self.earth_radius + altitude
        x = r * np.cos(M)
        y = r * np.sin(M) * np.cos(inclination)
        z = r * np.sin(M) * np.sin(inclination)
        
        # 应用升交点赤经旋转
        pos = np.array([
            x * np.cos(raan) - y * np.sin(raan),
            x * np.sin(raan) + y * np.cos(raan),
            z
        ])
        
        return pos
    
    def _check_line_of_sight(self, gs_pos: np.ndarray, sat_pos: np.ndarray) -> bool:
        """检查地面站和卫星之间是否有视线"""
        # 计算仰角
        elevation = self._calculate_elevation(gs_pos, sat_pos)
        
        # 检查最小仰角约束
        if elevation < self.min_elevation:
            return False
            
        # 检查地球遮挡
        gs_to_sat = sat_pos - gs_pos
        distance = np.linalg.norm(gs_to_sat)
        
        # 计算最近点到地心的距离
        p = np.dot(gs_pos, gs_to_sat) / distance
        closest_approach = np.linalg.norm(gs_pos + (p/distance)*gs_to_sat)
        
        return closest_approach >= self.earth_radius
    
    def _calculate_elevation(self, gs_pos: np.ndarray, sat_pos: np.ndarray) -> float:
        """计算卫星相对于地面站的仰角"""
        gs_to_sat = sat_pos - gs_pos
        zenith = gs_pos / np.linalg.norm(gs_pos)
        
        cos_elevation = np.dot(gs_to_sat, zenith) / (np.linalg.norm(gs_to_sat) * np.linalg.norm(zenith))
        elevation = np.degrees(np.arcsin(cos_elevation))
        
        return elevation 