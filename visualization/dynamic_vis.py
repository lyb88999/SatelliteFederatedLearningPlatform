import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.animation as animation
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class DynamicVisualizer:
    """动态可视化器"""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        
        # 创建地图
        self.ax = self.fig.add_subplot(111)
        self.map = Basemap(projection='mill',
                          lon_0=0,
                          resolution='l')
        
        # 绘制基础地图
        self.map.drawcoastlines()
        self.map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
        self.map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1])
        
        # 存储数据
        self.visibility_data: Dict[datetime, Dict[str, List[int]]] = {}
        self.ground_stations: Dict[str, Tuple[float, float]] = {}
        self.satellite_positions: Dict[datetime, Dict[int, List[Tuple[float, float, float]]]] = {}
        
        # 设置颜色映射
        self.orbit_colors = plt.cm.rainbow(np.linspace(0, 1, 6))  # 6个轨道的颜色
        
    def add_ground_station(self, station_id: str, lat: float, lon: float):
        """添加地面站"""
        self.ground_stations[station_id] = (lat, lon)
        
    def add_visibility_data(self, time: datetime, station_id: str, visible_orbits: List[int]):
        """添加可见性数据"""
        if time not in self.visibility_data:
            self.visibility_data[time] = {}
        self.visibility_data[time][station_id] = visible_orbits
        
    def add_satellite_positions(self, time: datetime, orbit_id: int, positions: List[Tuple[float, float, float]]):
        """添加卫星位置数据"""
        if time not in self.satellite_positions:
            self.satellite_positions[time] = {}
        self.satellite_positions[time][orbit_id] = positions
        
    def update_frame(self, frame_time: datetime):
        """更新单帧"""
        self.ax.clear()
        self.map.drawcoastlines()
        self.map.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
        self.map.drawmeridians(np.arange(-180,180,60),labels=[0,0,0,1])
        
        # 绘制卫星
        if frame_time in self.satellite_positions:
            for orbit_id, positions in self.satellite_positions[frame_time].items():
                color = self.orbit_colors[orbit_id]
                for pos in positions:
                    lat, lon = self._ecef_to_latlon(pos)
                    x, y = self.map(lon, lat)
                    self.map.plot(x, y, 'o', color=color, markersize=4, 
                                label=f'Orbit {orbit_id}' if pos is positions[0] else "")
        
        # 绘制地面站和覆盖范围
        for station_id, (lat, lon) in self.ground_stations.items():
            x, y = self.map(lon, lat)
            self.map.plot(x, y, 'ro', markersize=8, label=station_id)
            
            # 绘制覆盖圈和连接线
            if frame_time in self.visibility_data and station_id in self.visibility_data[frame_time]:
                visible_orbits = self.visibility_data[frame_time][station_id]
                if visible_orbits:
                    # 绘制覆盖圈
                    circle = plt.Circle((x, y), 1000000, color='blue', alpha=0.1,
                                     transform=self.ax.transData)
                    self.ax.add_patch(circle)
                    
                    # 绘制到可见卫星的连接线
                    if frame_time in self.satellite_positions:
                        for orbit_id in visible_orbits:
                            if orbit_id in self.satellite_positions[frame_time]:
                                sat_positions = self.satellite_positions[frame_time][orbit_id]
                                for sat_pos in sat_positions:
                                    sat_lat, sat_lon = self._ecef_to_latlon(sat_pos)
                                    sat_x, sat_y = self.map(sat_lon, sat_lat)
                                    self.ax.plot([x, sat_x], [y, sat_y], '--', 
                                               color=self.orbit_colors[orbit_id],
                                               alpha=0.3, linewidth=0.5)
        
        # 添加图例（去除重复项）
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), 
                      loc='upper right', fontsize=8)
        
        # 添加时间标签
        plt.title(f'Time: {frame_time.strftime("%Y-%m-%d %H:%M:%S")}')
        
    def animate(self, interval: int = 200):
        """创建动画"""
        times = sorted(self.visibility_data.keys())
        
        def update(frame):
            self.update_frame(times[frame])
            
        anim = animation.FuncAnimation(
            self.fig, 
            update,
            frames=len(times),
            interval=interval
        )
        
        return anim
    
    def save_animation(self, filename: str, fps: int = 5):
        """保存动画"""
        anim = self.animate(1000//fps)
        anim.save(filename, fps=fps)
        
    def show(self):
        """显示动画"""
        plt.show()
        
    def _ecef_to_latlon(self, pos: Tuple[float, float, float]) -> Tuple[float, float]:
        """ECEF坐标转经纬度"""
        x, y, z = pos
        r = np.sqrt(x*x + y*y)
        lon = np.degrees(np.arctan2(y, x))
        lat = np.degrees(np.arctan2(z, r))
        return lat, lon 