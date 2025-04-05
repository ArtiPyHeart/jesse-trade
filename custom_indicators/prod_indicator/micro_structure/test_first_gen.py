import unittest

import numpy as np
from first_gen import (
    bekker_parkinson_vol,
    corwin_schultz_estimator,
    roll_impact,
    roll_measure,
)


class TestFirstGen(unittest.TestCase):
    """市场微观结构指标测试类"""

    def setUp(self):
        """
        设置测试数据
        创建一个模拟的价格序列：包含一个明显的趋势和一些波动
        """
        n_candles = 100
        # 创建时间戳
        timestamps = np.arange(n_candles) * 60000  # 每分钟的毫秒数

        # 创建一个带趋势的价格序列
        base_price = 100 + np.linspace(0, 20, n_candles)  # 上涨趋势
        noise = np.random.normal(0, 1, n_candles)  # 随机波动

        # 生成OHLCV数据
        close = base_price + noise
        high = close + abs(np.random.normal(0, 0.5, n_candles))
        low = close - abs(np.random.normal(0, 0.5, n_candles))
        open = close + np.random.normal(0, 0.5, n_candles)
        volume = np.random.uniform(1000, 2000, n_candles)

        # 创建Jesse格式的K线数据 [timestamp, open, close, high, low, volume]
        self.candles = np.column_stack((timestamps, open, close, high, low, volume))

    def test_roll_measure(self):
        """测试Roll Measure指标"""
        # 测试非序列模式
        result = roll_measure(self.candles, window=20, sequential=False)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))

        # 测试序列模式
        result_seq = roll_measure(self.candles, window=20, sequential=True)
        self.assertEqual(len(result_seq), len(self.candles))
        self.assertTrue(
            np.all(np.isfinite(result_seq[20:]))
        )  # 窗口之后的值应该是有限的
        self.assertTrue(np.all(result_seq[:20] == 0))  # 窗口之前的值应该是0

    def test_roll_impact(self):
        """测试Roll Impact指标"""
        # 测试非序列模式
        result = roll_impact(self.candles, window=20, sequential=False)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))

        # 测试序列模式
        result_seq = roll_impact(self.candles, window=20, sequential=True)
        self.assertEqual(len(result_seq), len(self.candles))
        self.assertTrue(np.all(np.isfinite(result_seq[20:])))
        self.assertTrue(np.all(result_seq[:20] == 0))

        # 测试volume为0的情况
        test_candles = self.candles.copy()
        test_candles[:, 5] = 0  # 将volume设为0
        result_zero_vol = roll_impact(test_candles, window=20, sequential=True)
        self.assertTrue(
            np.all(np.isfinite(result_zero_vol))
        )  # 应该返回有限值（0或inf）

    def test_corwin_schultz_estimator(self):
        """测试Corwin-Schultz spread estimator指标"""
        # 测试非序列模式
        result = corwin_schultz_estimator(self.candles, window=20, sequential=False)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        self.assertTrue(0 <= result <= 1)  # spread应该在0到1之间

        # 测试序列模式
        result_seq = corwin_schultz_estimator(self.candles, window=20, sequential=True)
        self.assertEqual(len(result_seq), len(self.candles))
        self.assertTrue(
            np.all((result_seq >= 0) & (result_seq <= 1))
        )  # 所有值应该在0到1之间
        self.assertTrue(np.all(result_seq[:20] == 0))  # 窗口之前的值应该是0

    def test_bekker_parkinson_vol(self):
        """测试Bekker-Parkinson volatility指标"""
        # 测试非序列模式
        result = bekker_parkinson_vol(self.candles, window=20, sequential=False)
        self.assertIsInstance(result, float)
        self.assertTrue(np.isfinite(result))
        self.assertTrue(result >= 0)  # 波动率应该是非负的

        # 测试序列模式
        result_seq = bekker_parkinson_vol(self.candles, window=20, sequential=True)
        self.assertEqual(len(result_seq), len(self.candles))
        self.assertTrue(np.all(result_seq >= 0))  # 所有值应该是非负的
        self.assertTrue(np.all(result_seq[:20] == 0))  # 窗口之前的值应该是0

    def test_edge_cases(self):
        """测试边缘情况"""
        # 测试短数据
        short_candles = self.candles[:10]  # 小于窗口大小的数据

        # 所有指标都应该能处理短数据
        for func in [
            roll_measure,
            roll_impact,
            corwin_schultz_estimator,
            bekker_parkinson_vol,
        ]:
            # 非序列模式
            result = func(short_candles, window=20, sequential=False)
            self.assertTrue(np.isfinite(result))

            # 序列模式
            result_seq = func(short_candles, window=20, sequential=True)
            self.assertEqual(len(result_seq), len(short_candles))
            self.assertTrue(np.all(np.isfinite(result_seq)))

        # 测试极端价格
        extreme_candles = self.candles.copy()
        extreme_candles[:, 2] = 1e6  # 极大的收盘价
        extreme_candles[:, 3] = 1e6 + 100  # 极大的最高价
        extreme_candles[:, 4] = 1e6 - 100  # 极大的最低价

        # 所有指标都应该能处理极端价格
        for func in [
            roll_measure,
            roll_impact,
            corwin_schultz_estimator,
            bekker_parkinson_vol,
        ]:
            result = func(extreme_candles, window=20, sequential=True)
            self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
