import unittest
import numpy as np
from datetime import datetime
from flower.orbit_utils import OrbitCalculator, AdvancedOrbitCalculator
from flower.config import GroundStationConfig

class TestOrbitCalculator(unittest.TestCase):
    def setUp(self):
        self.calculator = OrbitCalculator(debug_mode=True)
        self.advanced_calculator = AdvancedOrbitCalculator(debug_mode=True)
        
        # 创建测试用的地面站
        self.test_station = GroundStationConfig(
            station_id="TestStation",
            latitude=40.0,
            longitude=116.0,
            max_range=2000.0,
            min_elevation=10.0,
            altitude=0.0
        )
        
    def test_visibility_check(self):
        """测试可见性检查"""
        # 测试基本轨道计算器
        for orbit_id in range(6):
            visibility = self.calculator.check_visibility(
                self.test_station, 
                orbit_id
            )
            self.assertIsInstance(visibility, bool)
            
        # 测试高级轨道计算器
        for orbit_id in range(6):
            visibility = self.advanced_calculator.check_visibility(
                self.test_station, 
                orbit_id
            )
            self.assertIsInstance(visibility, bool)
            
    def test_elevation_calculation(self):
        """测试仰角计算"""
        # 创建测试数据
        gs_pos = np.array([6371.0, 0.0, 0.0])  # 地面站在赤道上
        sat_pos = np.array([7151.0, 0.0, 0.0])  # 卫星在正上方780km
        
        # 计算仰角
        elevation = self.advanced_calculator._calculate_elevation(gs_pos, sat_pos)
        
        # 验证仰角在合理范围内
        self.assertGreaterEqual(elevation, -90)
        self.assertLessEqual(elevation, 90)
        
    def test_ground_station_position(self):
        """测试地面站位置计算"""
        position = self.advanced_calculator._get_ground_station_position(
            self.test_station.latitude,
            self.test_station.longitude,
            self.test_station.altitude
        )
        
        # 验证返回值是否为numpy数组
        self.assertIsInstance(position, np.ndarray)
        # 验证数组长度为3 (x, y, z)
        self.assertEqual(len(position), 3)
        # 验证位置在地球表面（考虑误差范围）
        distance_from_center = np.linalg.norm(position)
        self.assertAlmostEqual(distance_from_center, 6371.0, delta=1.0)

if __name__ == '__main__':
    unittest.main() 