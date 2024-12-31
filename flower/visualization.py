import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime, timedelta
from .config import SatelliteConfig, GroundStationConfig
from .orbit_utils import OrbitCalculator

class OrbitVisualizer:
    def __init__(self, orbit_calculator: OrbitCalculator):
        self.calculator = orbit_calculator
        self.earth_radius = orbit_calculator.earth_radius
        
    def plot_orbit(self, duration_hours=24, step_minutes=1):
        """绘制卫星轨道和地面站位置"""
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制地球
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
        y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
        z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
        
        # 计算并绘制卫星轨道
        current_time = datetime.now()
        orbit_x, orbit_y, orbit_z = [], [], []
        
        for minutes in range(duration_hours * 60):
            if minutes % step_minutes == 0:
                time = current_time + timedelta(minutes=minutes)
                pos = self.calculator._calculate_satellite_position(time)
                orbit_x.append(pos[0])
                orbit_y.append(pos[1])
                orbit_z.append(pos[2])
        
        ax.plot(orbit_x, orbit_y, orbit_z, 'r-', label='Satellite Orbit', alpha=0.5)
        
        # 绘制地面站
        for station in self.calculator.config.ground_stations:
            pos = self.calculator._calculate_ground_station_position(station, current_time)
            ax.scatter(pos[0], pos[1], pos[2], c='g', marker='^', s=100,
                      label=f'Ground Station: {station.station_id}')
            
            # 绘制覆盖范围（简化为圆锥）
            self._plot_coverage_cone(ax, station, current_time)
        
        # 设置图形属性
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title('Satellite Orbit and Ground Stations')
        ax.legend()
        
        # 设置坐标轴范围
        max_range = self.earth_radius + self.calculator.config.orbit_altitude + 1000
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        
        plt.show()
    
    def _plot_coverage_cone(self, ax, station: GroundStationConfig, time: datetime):
        """绘制地面站的覆盖范围"""
        gs_pos = self.calculator._calculate_ground_station_position(station, time)
        
        # 计算地面站的局部坐标系
        lat = np.radians(station.latitude)
        lon = np.radians(station.longitude)
        
        # 计算地面站的局部坐标轴
        up = np.array(gs_pos) / np.linalg.norm(gs_pos)  # 天顶方向
        east = np.array([-np.sin(lon), np.cos(lon), 0])  # 东向
        north = np.cross(up, east)  # 北向
        
        # 创建旋转矩阵
        R = np.vstack([north, east, up]).T
        
        # 计算覆盖圆锥
        min_elevation = np.radians(station.min_elevation)
        max_range = station.coverage_radius
        cone_height = max_range * np.sin(min_elevation)
        cone_radius = max_range * np.cos(min_elevation)
        
        # 创建圆锥顶点
        theta = np.linspace(0, 2*np.pi, 50)
        r = np.linspace(0, cone_radius, 20)
        theta, r = np.meshgrid(theta, r)
        
        # 计算圆锥表面点（在局部坐标系中）
        x_local = r * np.cos(theta)
        y_local = r * np.sin(theta)
        z_local = (cone_height * r) / cone_radius
        
        # 转换到全局坐标系
        points = np.stack([x_local.flatten(), y_local.flatten(), z_local.flatten()])
        transformed = np.dot(R, points)
        
        # 重塑数组并添加地面站位置偏移
        x_global = transformed[0].reshape(x_local.shape) + gs_pos[0]
        y_global = transformed[1].reshape(y_local.shape) + gs_pos[1]
        z_global = transformed[2].reshape(z_local.shape) + gs_pos[2]
        
        # 绘制覆盖锥面
        ax.plot_surface(x_global, y_global, z_global, alpha=0.1, color='green') 
    
    def animate_orbit(self, duration_hours=24):
        """动态显示卫星轨道"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.cla()  # 清除当前帧
            
            # 绘制地球
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = self.earth_radius * np.outer(np.cos(u), np.sin(v))
            y = self.earth_radius * np.outer(np.sin(u), np.sin(v))
            z = self.earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
            
            # 计算当前卫星位置
            current_time = datetime.now() + timedelta(minutes=frame)
            sat_pos = self.calculator._calculate_satellite_position(current_time)
            
            # 绘制卫星轨迹（过去10分钟）
            trail_x, trail_y, trail_z = [], [], []
            for i in range(max(0, frame-10), frame+1):
                time = datetime.now() + timedelta(minutes=i)
                pos = self.calculator._calculate_satellite_position(time)
                trail_x.append(pos[0])
                trail_y.append(pos[1])
                trail_z.append(pos[2])
            
            ax.plot(trail_x, trail_y, trail_z, 'r-', alpha=0.5)
            ax.scatter(sat_pos[0], sat_pos[1], sat_pos[2], c='red', s=100, label='Satellite')
            
            # 绘制地面站和覆盖范围
            for station in self.calculator.config.ground_stations:
                pos = self.calculator._calculate_ground_station_position(station, current_time)
                ax.scatter(pos[0], pos[1], pos[2], c='g', marker='^', s=100,
                          label=f'Ground Station: {station.station_id}')
                
                # 检查可见性并绘制连线
                if self.calculator.calculate_visibility(station, current_time):
                    ax.plot([pos[0], sat_pos[0]], [pos[1], sat_pos[1]], [pos[2], sat_pos[2]],
                           'g--', alpha=0.5)
                
                self._plot_coverage_cone(ax, station, current_time)
            
            # 设置视图
            ax.set_xlim([-self.earth_radius*2, self.earth_radius*2])
            ax.set_ylim([-self.earth_radius*2, self.earth_radius*2])
            ax.set_zlim([-self.earth_radius*2, self.earth_radius*2])
            ax.set_title(f'Time: {current_time.strftime("%H:%M:%S")}')
        
        from matplotlib.animation import FuncAnimation
        anim = FuncAnimation(fig, update, frames=duration_hours*60, interval=50)
        plt.show() 