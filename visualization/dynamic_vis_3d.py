import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class Dynamic3DVisualizer:
    """3D动态可视化器"""
    
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 存储数据
        self.visibility_data: Dict[datetime, Dict[str, List[int]]] = {}
        self.ground_stations: Dict[str, Tuple[float, float]] = {}
        self.satellite_positions: Dict[datetime, Dict[int, List[Tuple[float, float, float]]]] = {}
        
        # 设置颜色
        self.orbit_colors = {
            0: 'red',
            1: 'blue',
            2: 'green',
            3: 'purple',
            4: 'orange',
            5: 'brown'
        }
        
        # 绘制地球
        self._draw_earth()
        
    def _draw_earth(self):
        """绘制3D地球"""
        # 创建球体网格
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 6371 * np.outer(np.cos(u), np.sin(v))
        y = 6371 * np.outer(np.sin(u), np.sin(v))
        z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # 绘制地球表面
        self.ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # 绘制经纬线
        for lat in range(-90, 91, 30):
            self._draw_parallel(lat)
        for lon in range(-180, 181, 30):
            self._draw_meridian(lon)
    
    def _draw_parallel(self, lat):
        """绘制纬线"""
        lat_rad = np.radians(lat)
        theta = np.linspace(0, 2*np.pi, 100)
        r = 6371 * np.cos(lat_rad)
        z = 6371 * np.sin(lat_rad)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        self.ax.plot(x, y, z, 'k:', alpha=0.2)
    
    def _draw_meridian(self, lon):
        """绘制经线"""
        lon_rad = np.radians(lon)
        phi = np.linspace(-np.pi/2, np.pi/2, 100)
        r = 6371
        x = r * np.cos(phi) * np.cos(lon_rad)
        y = r * np.cos(phi) * np.sin(lon_rad)
        z = r * np.sin(phi)
        self.ax.plot(x, y, z, 'k:', alpha=0.2)
    
    def _latlon_to_xyz(self, lat: float, lon: float, alt: float = 0) -> Tuple[float, float, float]:
        """将经纬度转换为3D坐标"""
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        r = 6371 + alt
        x = r * np.cos(lat_rad) * np.cos(lon_rad)
        y = r * np.cos(lat_rad) * np.sin(lon_rad)
        z = r * np.sin(lat_rad)
        return x, y, z
    
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
        self.ax.cla()
        self._draw_earth()
        
        # 设置视角
        self.ax.view_init(elev=20, azim=frame_time.hour * 15)
        
        # 绘制卫星和轨道
        if frame_time in self.satellite_positions:
            for orbit_id, positions in self.satellite_positions[frame_time].items():
                color = self.orbit_colors[orbit_id]
                xs, ys, zs = zip(*positions)
                # 绘制轨道线
                self.ax.plot(xs, ys, zs, '--', color=color, alpha=0.3)
                # 绘制卫星
                self.ax.scatter(xs, ys, zs, c=color, marker='o', s=50)
        
        # 绘制地面站和连接线
        for station_id, (lat, lon) in self.ground_stations.items():
            x, y, z = self._latlon_to_xyz(lat, lon)
            self.ax.scatter(x, y, z, c='red', marker='^', s=100)
            
            # 绘制可见连接
            if (frame_time in self.visibility_data and 
                station_id in self.visibility_data[frame_time]):
                visible_orbits = self.visibility_data[frame_time][station_id]
                for orbit_id in visible_orbits:
                    if orbit_id in self.satellite_positions[frame_time]:
                        sat_positions = self.satellite_positions[frame_time][orbit_id]
                        for sat_pos in sat_positions:
                            self.ax.plot([x, sat_pos[0]], [y, sat_pos[1]], [z, sat_pos[2]], 
                                       '-', color=self.orbit_colors[orbit_id], alpha=0.5)
        
        # 设置标题和轴标签
        self.ax.set_title(f'Time: {frame_time.strftime("%Y-%m-%d %H:%M:%S")}')
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        
        # 设置轴范围
        limit = 8000
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([-limit, limit])
    
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
        print(f"正在生成3D动画 {filename}...")
        anim = self.animate(1000//fps)
        anim.save(filename, fps=fps)
        print(f"动画已保存到 {filename}")
    
    def show(self):
        """显示动画"""
        plt.show() 