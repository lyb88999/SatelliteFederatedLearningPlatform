import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

def calculate_satellite_position(a, i, raan, arg_perigee):
    """计算卫星位置"""
    # 简化的开普勒轨道
    x = a * np.cos(arg_perigee)
    y = a * np.sin(arg_perigee) * np.cos(i)
    z = a * np.sin(arg_perigee) * np.sin(i)
    
    # 应用升交点赤经旋转
    pos = np.array([
        x * np.cos(raan) - y * np.sin(raan),
        x * np.sin(raan) + y * np.cos(raan),
        z
    ])
    return pos

def visualize_constellation(earth_radius=6371.0):
    """可视化星座拓扑"""
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # 画出地球
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    
    # 铱星星座参数
    num_planes = 6
    sats_per_plane = 11
    altitude = 780.0
    inclination = np.radians(86.4)
    
    # 画出轨道面和卫星
    colors = plt.cm.rainbow(np.linspace(0, 1, num_planes))
    for plane_id in range(num_planes):
        raan = np.radians(plane_id * 360.0 / num_planes)
        orbit_points = []
        
        # 画出轨道
        theta = np.linspace(0, 2*np.pi, 100)
        for t in theta:
            pos = calculate_satellite_position(
                earth_radius + altitude,
                inclination,
                raan,
                t
            )
            orbit_points.append(pos)
        orbit_points = np.array(orbit_points)
        ax.plot(orbit_points[:, 0], orbit_points[:, 1], orbit_points[:, 2], 
                '-', color=colors[plane_id], alpha=0.3)
        
        # 画出卫星
        for sat_id in range(sats_per_plane):
            arg_perigee = np.radians(sat_id * 360.0 / sats_per_plane)
            pos = calculate_satellite_position(
                earth_radius + altitude,
                inclination,
                raan,
                arg_perigee
            )
            ax.scatter(*pos, c=[colors[plane_id]], marker='o', s=100)
    
    # 画出地面站
    ground_stations = [
        ("Beijing", 39.9042, 116.4074),
        ("NewYork", 40.7128, -74.0060),
        ("London", 51.5074, -0.1278),
        ("Sydney", -33.8688, 151.2093),
        ("Moscow", 55.7558, 37.6173),
        ("SaoPaulo", -23.5505, -46.6333)
    ]
    
    for name, lat, lon in ground_stations:
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = earth_radius * np.sin(lat_rad)
        ax.scatter(x, y, z, c='g', marker='^', s=200)
        ax.text(x*1.1, y*1.1, z*1.1, name)
    
    # 设置图形属性
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Iridium Constellation Topology')
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    # 保存图片
    plt.savefig('results/constellation_topology.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_visibility(earth_radius=6371.0):
    """可视化地面站可见性"""
    fig = plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    # 画出地面站和可见区域
    ground_stations = [
        ("Beijing", 39.9042, 116.4074),
        ("NewYork", 40.7128, -74.0060),
        ("London", 51.5074, -0.1278),
        ("Sydney", -33.8688, 151.2093),
        ("Moscow", 55.7558, 37.6173),
        ("SaoPaulo", -23.5505, -46.6333)
    ]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(ground_stations)))
    
    for (name, lat, lon), color in zip(ground_stations, colors):
        # 画出地面站点
        plt.scatter(lon, lat, c=[color], marker='^', s=200, label=name)
        
        # 画出可见区域（简化模型）
        if abs(lat) > 60:
            visible_orbits = [0, 5]
        elif abs(lat) > 30:
            visible_orbits = [1, 2, 3]
        else:
            visible_orbits = [2, 3, 4]
            
        # 在图例中显示可见轨道
        plt.text(lon+5, lat+5, f"Visible orbits: {visible_orbits}", 
                color=color[:-1], fontsize=8)
    
    # 画出轨道面投影
    for plane_id in range(6):
        lon = plane_id * 60 - 180
        plt.axvline(x=lon, color='gray', linestyle='--', alpha=0.3)
        plt.text(lon, 80, f"Orbit {plane_id}", rotation=90)
    
    # 设置图形属性
    plt.grid(True, alpha=0.3)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Ground Station Visibility')
    plt.legend()
    
    # 设置坐标轴范围
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
    # 保存图片
    plt.savefig('results/visibility_map.png', dpi=300, bbox_inches='tight')
    plt.close() 