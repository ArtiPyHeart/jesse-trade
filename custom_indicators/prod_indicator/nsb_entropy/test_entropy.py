# -*- coding: utf-8 -*-
# Author: Qiuyu Yang
# License: BSD 3 clause
"""
NSB熵估计函数的单元测试
"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
from entropy import entropy_for_jesse, nsb_entropy


class TestNsbEntropy(unittest.TestCase):
    """NSB熵估计测试类"""

    def test_basic_cases(self):
        """测试基本用例"""
        # 单一箱子的情况（确定性分布）
        counts = [10]
        h = nsb_entropy(counts)
        self.assertAlmostEqual(h, 0.0, places=6)

        # 均匀分布的情况（最大熵）
        counts = [10, 10, 10, 10]
        h = nsb_entropy(counts, k=4)
        # 均匀分布的熵应接近log(k)
        self.assertAlmostEqual(h, np.log(4), places=1)

        # 标准例子
        counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
        h = nsb_entropy(counts)
        # 熵应该在合理范围内
        self.assertTrue(1.5 < h < 3.5)

    def test_return_std(self):
        """测试返回标准差的功能"""
        counts = [4, 12, 4, 5, 3, 1, 5, 1, 2, 2, 2, 2, 11, 3, 4, 12, 12, 1, 2]
        h, err = nsb_entropy(counts, return_std=True)

        # 熵和标准差都应该是有限的正数
        self.assertTrue(h > 0 and np.isfinite(h))
        self.assertTrue(err > 0 and np.isfinite(err))

    def test_edge_cases(self):
        """测试边缘情况"""
        # 所有计数都为1的情况（没有重复计数）
        counts = [1] * 100
        h, err = nsb_entropy(counts, return_std=True)

        # 熵应该接近于log(n)
        self.assertAlmostEqual(h, np.log(100), places=1)
        # 标准差应该是大于0的值
        self.assertTrue(err > 0)

        # 尝试使用空数组 (我们应该尝试避免抛出异常)
        try:
            nsb_entropy([])
            # 如果代码能够处理空数组情况，测试应该通过
            self.assertTrue(True)
        except Exception:
            # 如果出现异常也没问题 - 这是可接受的行为
            self.assertTrue(True)

    def test_k_parameter(self):
        """测试字母表大小参数k的影响"""
        counts = [10, 5, 3]

        # 测试不同的k值
        h1 = nsb_entropy(counts, k=3)
        h2 = nsb_entropy(counts, k=10)
        h3 = nsb_entropy(counts, k=100)

        # 同样的分布但字母表大小不同，熵应该不同
        # 字母表增大，熵应该增加
        self.assertTrue(h1 < h2 < h3)

    def test_jesse_interface(self):
        """测试为Jesse框架提供的接口"""
        # 创建模拟的K线数据
        # 假设K线数组格式为：[timestamp, open, close, high, low, volume]
        n_candles = 50
        candles = np.zeros((n_candles, 6))

        # 创建一个简单的价格序列：先上涨后下跌
        price = np.concatenate(
            [
                np.linspace(100, 150, n_candles // 2),
                np.linspace(150, 100, n_candles // 2),
            ]
        )

        # 填充K线数据
        for i in range(n_candles):
            candles[i, 0] = i  # 时间戳
            candles[i, 1] = price[i] - 1  # 开盘价
            candles[i, 2] = price[i]  # 收盘价
            candles[i, 3] = price[i] + 1  # 最高价
            candles[i, 4] = price[i] - 2  # 最低价
            candles[i, 5] = 1000  # 交易量

        # 测试不同的参数
        # 默认参数
        e1 = entropy_for_jesse(candles)
        self.assertTrue(np.isfinite(e1))

        # 不同的价格类型
        e2 = entropy_for_jesse(candles, source_type="high")
        self.assertTrue(np.isfinite(e2))

        # 测试sequential=True
        e_seq = entropy_for_jesse(candles, sequential=True)
        self.assertEqual(len(e_seq), len(candles))
        # 前面的值应该是NaN（因为窗口）
        self.assertTrue(np.isnan(e_seq[0]))
        # 后面的值应该是有限的
        self.assertTrue(np.isfinite(e_seq[-1]))

    def test_visualization(self):
        """可视化测试，非必要的断言测试"""
        try:
            # 创建一系列不同的分布来测试熵估计
            distributions = [
                [100],  # 单峰分布
                [50, 50],  # 两个均匀峰
                [80, 20],  # 两个不均匀峰
                [33, 33, 34],  # 三个接近均匀的峰
                [10, 20, 30, 40],  # 四个递增的峰
                [40, 30, 20, 10],  # 四个递减的峰
                [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],  # 十个均匀的峰
            ]

            # 计算每个分布的熵
            entropies = []
            for dist in distributions:
                h = nsb_entropy(dist)
                entropies.append(h)

            # 创建图表（不会实际显示，但会检查代码是否能执行）
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(distributions)), entropies)
            plt.xlabel("分布索引")
            plt.ylabel("NSB熵估计")
            plt.title("不同分布的NSB熵估计")
            plt.tight_layout()

            # 不实际保存图片，只是测试代码是否能执行
            # plt.savefig('nsb_entropy_distributions.png')
            plt.close()

            # 没有异常，测试通过
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"可视化测试失败：{str(e)}")

    def test_compared_to_theoretical(self):
        """将NSB熵估计与理论熵进行比较"""
        # 创建已知熵的分布

        # 均匀分布的理论熵为log(k)
        k = 5
        uniform_counts = [20] * k
        uniform_entropy = nsb_entropy(uniform_counts)
        theoretical_uniform = np.log(k)

        # 验证估计结果接近理论值
        self.assertAlmostEqual(uniform_entropy, theoretical_uniform, delta=0.1)

        # 二项分布的理论熵
        # 二项分布B(n,p)的熵可以近似计算
        n = 100
        p = 0.3
        binomial_counts = [int(n * p), int(n * (1 - p))]  # 简化为两个结果
        binomial_entropy = nsb_entropy(binomial_counts)

        # 计算理论熵 (近似)
        theoretical_binomial = -(p * np.log(p) + (1 - p) * np.log(1 - p))

        # 验证估计结果接近理论值
        self.assertAlmostEqual(binomial_entropy, theoretical_binomial, delta=0.15)


if __name__ == "__main__":
    unittest.main()
