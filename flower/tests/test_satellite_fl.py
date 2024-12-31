import asyncio
from datetime import datetime, timedelta
import numpy as np
from flower.config import GroundStationConfig, SatelliteConfig
from flower.orbit_utils import OrbitCalculator
from flower.visualization import OrbitVisualizer

async def test_communication_windows():
    # 创建地面站
    ground_stations = [
        GroundStationConfig(
            "Beijing", 
            39.9042, 
            116.4074, 
            coverage_radius=2000,
            min_elevation=10.0
        ),
        GroundStationConfig(
            "Shanghai", 
            31.2304, 
            121.4737, 
            coverage_radius=2000,
            min_elevation=10.0
        ),
        GroundStationConfig(
            "Guangzhou", 
            23.1291, 
            113.2644, 
            coverage_radius=2000,
            min_elevation=10.0
        )
    ]

    # 创建卫星配置
    satellite_config = SatelliteConfig(
        orbit_altitude=550.0,
        orbit_inclination=97.6,
        orbital_period=95,
        ground_stations=ground_stations,
        ascending_node=0.0,
        mean_anomaly=0.0
    )

    # 创建轨道��算器
    orbit_calculator = OrbitCalculator(satellite_config)

    # 模拟24小时的卫星运行
    current_time = datetime.now()
    print("卫星可见性时间窗口:")
    print("-" * 50)
    
    windows = {station.station_id: {"start": None, "last_visible": False} 
              for station in ground_stations}
    
    for minutes in range(0, 24 * 60):
        test_time = current_time + timedelta(minutes=minutes)
        
        # 每小时打印一次卫星位置
        if minutes % 60 == 0:
            sat_pos = orbit_calculator._calculate_satellite_position(test_time)
            sat_height = np.sqrt(np.sum(np.square(sat_pos))) - orbit_calculator.earth_radius
            print(f"\n时间: {test_time.strftime('%H:%M:%S')}")
            print(f"卫星高度: {sat_height:.1f}km")
            print(f"卫星位置: X={sat_pos[0]:.1f}, Y={sat_pos[1]:.1f}, Z={sat_pos[2]:.1f}")
        
        # 检查每个地面站的可见性
        for station in ground_stations:
            is_visible = orbit_calculator.calculate_visibility(station, test_time)
            
            # 如果是新的可见性窗口开始
            if is_visible and not windows[station.station_id]["last_visible"]:
                windows[station.station_id]["start"] = test_time
                print(f"\n地面站: {station.station_id}")
                print(f"开始时间: {test_time.strftime('%H:%M:%S')}")
            
            # 如果可见性窗口结束
            elif not is_visible and windows[station.station_id]["last_visible"]:
                start_time = windows[station.station_id]["start"]
                duration = (test_time - start_time).total_seconds() / 60
                print(f"结束时间: {test_time.strftime('%H:%M:%S')}")
                print(f"通过持续时间: {duration:.1f}分钟")
                print("-" * 50)
            
            windows[station.station_id]["last_visible"] = is_visible

    # 添加可视化
    visualizer = OrbitVisualizer(orbit_calculator)
    visualizer.plot_orbit(duration_hours=24, step_minutes=1)  # 静态轨道图
    visualizer.animate_orbit(duration_hours=2)  # 动态轨道图（2小时）

if __name__ == "__main__":
    asyncio.run(test_communication_windows()) 