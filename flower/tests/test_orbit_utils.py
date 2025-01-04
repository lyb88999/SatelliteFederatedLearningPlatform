import pytest
from datetime import datetime, timedelta
import numpy as np
from flower.config import SatelliteConfig, GroundStationConfig
from flower.orbit_utils import OrbitCalculator

def test_basic_orbit_utils():
    """测试基本轨道计算功能"""
    # 创建轨道计算器
    orbit_calculator = OrbitCalculator(debug_mode=True)
    earth_radius = orbit_calculator.earth_radius
    
    # 创建卫星配置
    satellites = []
    for orbit_id in range(2):  # 2个轨道面
        raan = orbit_id * 180.0  # 轨道面均匀分布
        for sat_id in range(2):  # 每个轨道2颗卫星
            phase_angle = sat_id * 180.0  # 卫星在轨道内均匀分布
            satellites.append(
                SatelliteConfig(
                    orbit_id=orbit_id,
                    sat_id=len(satellites),
                    semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
                    eccentricity=0.001,
                    inclination=97.6,
                    raan=raan,
                    arg_perigee=phase_angle,
                    epoch=datetime.now()
                )
            )
    
    # 创建地面站
    ground_stations = []
    locations = [
        ("Beijing", 39.9042, 116.4074),
        ("NewYork", 40.7128, -74.0060)
    ]
    
    for name, lat, lon in locations:
        config = GroundStationConfig(
            station_id=name,
            latitude=lat,
            longitude=lon,
            max_range=2000.0,
            min_elevation=10.0,
            max_satellites=4
        )
        ground_stations.append(config)
    
    # 测试卫星位置计算
    current_time = datetime.now()
    print("\n卫星位置计算测试:")
    for satellite in satellites:
        position = orbit_calculator.calculate_satellite_position(satellite, current_time)
        height = np.sqrt(np.sum(np.square(position))) - earth_radius
        print(f"\n卫星 {satellite.sat_id} (轨道 {satellite.orbit_id}):")
        print(f"位置: X={position[0]:.1f}, Y={position[1]:.1f}, Z={position[2]:.1f}")
        print(f"高度: {height:.1f}km")
        
        # 验证高度
        assert abs(height - 550.0) < 1.0, "卫星高度应该在550km左右"
    
    # 测试可见性计算
    print("\n可见性计算测试:")
    visibility_map = {
        "Beijing": [0],     # 北京只能看到轨道0的卫星
        "NewYork": [1]      # 纽约只能看到轨道1的卫星
    }
    
    for station in ground_stations:
        station_visible = []
        for satellite in satellites:
            if satellite.orbit_id in visibility_map[station.station_id]:
                if orbit_calculator.check_satellite_visibility(
                    satellite, station, current_time):
                    station_visible.append(satellite.sat_id)
        print(f"\n地面站 {station.station_id} 可见卫星: {station_visible}")
        
        # 验证可见性
        visible_sats = [sat for sat in satellites 
                       if sat.orbit_id in visibility_map[station.station_id]]
        assert len(station_visible) <= len(visible_sats), \
            f"{station.station_id} 不应该看到超过{len(visible_sats)}颗卫星"

def test_advanced_orbit_utils():
    """测试高级轨道计算功能"""
    calculator = OrbitCalculator(debug_mode=True)
    earth_radius = calculator.earth_radius
    current_time = datetime.now()
    
    # 创建测试卫星
    sat1 = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,
        eccentricity=0.001,
        inclination=97.6,
        raan=0.0,
        arg_perigee=0.0,
        epoch=current_time
    )
    
    sat2 = SatelliteConfig(
        orbit_id=0,
        sat_id=1,
        semi_major_axis=earth_radius + 550.0,
        eccentricity=0.001,
        inclination=97.6,
        raan=0.0,
        arg_perigee=180.0,  # 对面的卫星
        epoch=current_time
    )
    
    # 测试轨道周期
    period = calculator.calculate_orbital_period(sat1.semi_major_axis)
    print(f"\n轨道周期: {period/60:.2f} 分钟")
    assert 90 < period/60 < 100  # LEO轨道周期约为90-100分钟
    
    # 测试卫星间距离
    distance = calculator.calculate_satellite_distance(sat1, sat2, current_time)
    print(f"卫星间距离: {distance:.2f} km")
    assert distance > 0
    
    # 测试地球遮挡
    pos1 = calculator.calculate_satellite_position(sat1, current_time)
    pos2 = calculator.calculate_satellite_position(sat2, current_time)
    # 转换为 numpy 数组
    pos1_array = np.array(pos1)
    pos2_array = np.array(pos2)
    is_obstructed = calculator._check_earth_obstruction(pos1_array, pos2_array)
    assert is_obstructed, "对面的卫星应该被地球遮挡"

def test_satellite_daily_positions():
    """测试卫星在一天中不同时刻的位置"""
    calculator = OrbitCalculator(debug_mode=True)
    earth_radius = calculator.earth_radius
    
    # 创建一个测试卫星
    start_time = datetime(2024, 1, 5, 0, 0, 0)  # 从2024年1月5日0点开始
    test_satellite = SatelliteConfig(
        orbit_id=0,
        sat_id=0,
        semi_major_axis=earth_radius + 550.0,  # 550km轨道高度
        eccentricity=0.001,
        inclination=97.6,  # 太阳同步轨道
        raan=0.0,
        arg_perigee=0.0,
        epoch=start_time
    )
    
    # 计算轨道周期
    period_minutes = calculator.calculate_orbital_period(test_satellite.semi_major_axis) / 60
    print(f"\n卫星轨道周期: {period_minutes:.2f} 分钟")
    
    # 在一天中每10分钟采样一次
    time_points = [
        start_time + timedelta(minutes=i*10) 
        for i in range(145)  # 24小时 = 144个10分钟间隔
    ]
    
    print("\n卫星在一天中的位置变化（每小时第一个采样点）:")
    print("时间\t\t\t经度\t纬度\t高度(km)\t速度(km/s)")
    
    # 存储轨迹点用于可视化
    longitudes = []
    latitudes = []
    
    for i, time in enumerate(time_points):
        # 计算卫星位置
        position = calculator.calculate_satellite_position(test_satellite, time)
        
        # 将笛卡尔坐标转换为地理坐标
        x, y, z = position
        r = np.sqrt(x*x + y*y + z*z)
        lat = np.arcsin(z/r) * 180/np.pi
        lon = np.arctan2(y, x) * 180/np.pi
        height = r - earth_radius
        
        # 计算速度（轨道速度）
        velocity = np.sqrt(calculator.earth_mu / r)  # km/s
        
        # 只打印每小时的第一个采样点
        if i % 6 == 0:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t"
                  f"{lon:6.2f}°\t{lat:6.2f}°\t{height:8.2f}\t{velocity:6.2f}")
        
        # 存储轨迹点
        longitudes.append(lon)
        latitudes.append(lat)
        
        # 验证高度保持在合理范围内
        assert 540 <= height <= 560, f"卫星高度 {height:.2f}km 超出预期范围"
        # 验证速度在LEO卫星的典型范围内（约7.5-7.8 km/s）
        assert 7.4 <= velocity <= 7.9, f"卫星速度 {velocity:.2f}km/s 超出预期范围"
    
    # 使用matplotlib绘制地面轨迹
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(20, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # 添加自然地球底图
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.3)
        ax.gridlines(draw_labels=True, dms=True, alpha=0.2)
        
        # 绘制卫星轨迹
        # 将轨迹点分段，避免跨越180度经线的连线
        track_segments = []
        current_segment = []
        
        for i in range(len(longitudes)):
            lon = longitudes[i]
            if lon > 180:
                lon -= 360
            elif lon < -180:
                lon += 360
            
            if current_segment and abs(lon - current_segment[-1][0]) > 180:
                # 当跨越180度经线时，开始新的段
                track_segments.append(current_segment)
                current_segment = []
            
            current_segment.append((lon, latitudes[i]))
            
        if current_segment:
            track_segments.append(current_segment)
        
        # 绘制每一段轨迹
        for segment in track_segments:
            lons, lats = zip(*segment)
            plt.plot(lons, lats,
                    color='blue',
                    linewidth=2,
                    alpha=0.6,
                    transform=ccrs.PlateCarree(),
                    zorder=2)
        
        # 添加轨迹图例（只添加一次）
        plt.plot([], [], 
                color='blue',
                linewidth=2,
                alpha=0.6,
                label='Satellite Track')
        
        # 每小时添加一个位置点和时间标记
        for i in range(0, len(time_points), 6):
            lon = longitudes[i]
            if lon > 180:
                lon -= 360
            elif lon < -180:
                lon += 360
                
            plt.scatter(lon, latitudes[i],
                       color='red',
                       s=30,
                       transform=ccrs.PlateCarree(),
                       zorder=3)
            if i % 12 == 0:  # 每2小时添加时间标记
                plt.annotate(f"{time_points[i].strftime('%H:%M')}", 
                            xy=(lon, latitudes[i]),
                            xytext=(5, 5), textcoords='offset points',
                            color='darkblue',
                            fontsize=8,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                            transform=ccrs.PlateCarree(),
                            zorder=4)
        
        # 标记起点
        start_lon = longitudes[0]
        if start_lon > 180:
            start_lon -= 360
        elif start_lon < -180:
            start_lon += 360
            
        plt.scatter(start_lon, latitudes[0], 
                   color='green', s=100, 
                   label='Start Point', 
                   transform=ccrs.PlateCarree(),
                   zorder=5)
        
        # 添加图例
        plt.legend(loc='upper right')
        
        # 设置标题和标签
        plt.title('Satellite Ground Track (24 Hours)\n'
                 f'Orbit Height: 550km, Inclination: 97.6°, Period: {period_minutes:.1f}min\n'
                 'Sampling Interval: 10min',
                 pad=20)
        
        # 保存高清图片
        plt.savefig('satellite_ground_track.png', dpi=300, bbox_inches='tight')
        print("\n地面轨迹图已保存为 satellite_ground_track.png")
        
    except ImportError as e:
        print(f"\n注意：缺少可视化所需的库（{str(e)}），请安装 matplotlib 和 cartopy")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 