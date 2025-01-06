import asyncio
from experiments.iridium_simulation import run_simulation, create_ground_stations
from visualization.constellation_vis import visualize_constellation, visualize_visibility
from flower.orbit_utils import OrbitCalculator, AdvancedOrbitCalculator
import torch
import os

async def test_federated_learning(use_grouping=False, use_advanced_orbit=False):
    """运行联邦学习测试
    Args:
        use_grouping: 是否使用分组训练模式
        use_advanced_orbit: 是否使用高级轨道计算器
    """
    # 确保目录存在
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 生成可视化
    print("生成星座拓扑可视化...")
    visualize_constellation()
    visualize_visibility()
    
    print("开始联邦学习测试...")
    try:
        await run_simulation(
            use_grouping=use_grouping,
            use_advanced_orbit=use_advanced_orbit
        )
        print("测试完成!")
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise e

async def test_visibility(use_advanced_orbit=True):
    """测试地面站-卫星可见性"""
    print("\n=== 可见性测试 ===")
    
    # 创建轨道计算器
    if use_advanced_orbit:
        orbit_calculator = AdvancedOrbitCalculator(debug_mode=True)
        print("使用高级轨道计算器")
    else:
        orbit_calculator = OrbitCalculator(debug_mode=True)
        print("使用简化轨道计算器")
    
    # 创建地面站
    ground_stations = await create_ground_stations(orbit_calculator)
    
    # 测试每个地面站的可见性
    for station in ground_stations:
        print(f"\n地面站: {station.station_id} ({station.latitude}°N, {station.longitude}°E)")
        visible_orbits = []
        
        # 检查每个轨道的可见性
        for orbit_id in range(6):
            if orbit_calculator.check_visibility(station, orbit_id):
                visible_orbits.append(orbit_id)
                
        print(f"可见轨道: {visible_orbits}")
        print(f"可见轨道数: {len(visible_orbits)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-grouping', action='store_true', 
                       help='是否使用分组训练模式')
    parser.add_argument('--use-advanced-orbit', action='store_true', 
                       help='是否使用高级轨道计算器')
    parser.add_argument('--test-visibility', action='store_true',
                       help='是否只测试可见性')
    parser.add_argument('--dynamic-simulation', action='store_true',
                       help='是否运行动态时间仿真')
    args = parser.parse_args()
    
    if args.dynamic_simulation:
        from experiments.dynamic_simulation import test_dynamic_simulation
        asyncio.run(test_dynamic_simulation())
    elif args.test_visibility:
        asyncio.run(test_visibility(use_advanced_orbit=args.use_advanced_orbit))
    else:
        asyncio.run(test_federated_learning(
            use_grouping=args.use_grouping,
            use_advanced_orbit=args.use_advanced_orbit
        )) 