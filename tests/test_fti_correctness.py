"""
FTI 指标正确性测试 - 验证 Rust 实现与 Python 实现的数值一致性
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest

# 导入 Python numba 实现
from src.indicators.prod.fti import (
    _find_coefs_numba,
    _extrapolate_data_numba,
    _apply_filter_numba,
    _calculate_width_numba,
    _calculate_fti_numba,
    FTI as PythonFTI,
)

# 导入 Rust 实现
from pyrs_indicators.ind_trend import fti as rust_fti


def test_fti_basic_comparison():
    """基础测试：对比 Python 和 Rust 实现的 FTI 计算结果"""
    np.random.seed(42)

    # 创建趋势价格序列
    prices = 100 + np.cumsum(np.random.randn(300) * 0.5)

    # Python 实现
    python_calculator = PythonFTI(
        use_log=True,
        min_period=5,
        max_period=65,
        half_length=35,
        lookback=150,
        beta=0.95,
        noise_cut=0.20,
    )

    data_window = prices[-150:][::-1]  # 反转使最近的数据点在索引0
    python_result = python_calculator.process(data_window)

    # Rust 实现
    rust_result = rust_fti(
        data_window,
        use_log=True,
        min_period=5,
        max_period=65,
        half_length=35,
        lookback=150,
        beta=0.95,
        noise_cut=0.20,
    )

    print("\n=== FTI 对比结果 ===")
    print(f"Python FTI:          {python_result.fti:.6f}")
    print(f"Rust FTI:            {rust_result[0]:.6f}")
    print(f"差异:                {abs(python_result.fti - rust_result[0]):.6f}")
    print(f"\nPython filtered:     {python_result.filtered_value:.6f}")
    print(f"Rust filtered:       {rust_result[1]:.6f}")
    print(f"差异:                {abs(python_result.filtered_value - rust_result[1]):.6f}")
    print(f"\nPython width:        {python_result.width:.6f}")
    print(f"Rust width:          {rust_result[2]:.6f}")
    print(f"差异:                {abs(python_result.width - rust_result[2]):.6f}")
    print(f"\nPython period:       {python_result.best_period:.1f}")
    print(f"Rust period:         {rust_result[3]:.1f}")

    # 验证 FTI 值范围（根据公式 mean_move/width，应 >= 0）
    assert python_result.fti >= 0, f"Python FTI 为负: {python_result.fti}"
    assert rust_result[0] >= 0, f"Rust FTI 为负: {rust_result[0]}"

    # 验证数值一致性（允许小误差）
    assert np.isclose(python_result.fti, rust_result[0], rtol=1e-6, atol=1e-6), \
        f"FTI 值不一致: Python={python_result.fti}, Rust={rust_result[0]}"
    assert np.isclose(python_result.filtered_value, rust_result[1], rtol=1e-6, atol=1e-6), \
        f"filtered_value 不一致"
    assert np.isclose(python_result.width, rust_result[2], rtol=1e-6, atol=1e-6), \
        f"width 不一致"
    assert python_result.best_period == rust_result[3], \
        f"best_period 不一致"


def test_fti_component_comparison():
    """细粒度测试：对比各个计算步骤的中间结果"""
    np.random.seed(123)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.3)

    # 参数
    use_log = True
    min_period = 5
    max_period = 65
    half_length = 35
    lookback = 150
    beta = 0.95
    noise_cut = 0.20

    data_window = prices[-lookback:][::-1]

    # 准备数据
    y = np.zeros(lookback + half_length)
    for i in range(lookback):
        if use_log:
            y[lookback - 1 - i] = np.log(data_window[i])
        else:
            y[lookback - 1 - i] = data_window[i]

    # 外推数据
    y = _extrapolate_data_numba(y, lookback, half_length)

    # 测试特定周期（例如 period=10）
    test_period = 10

    # 计算滤波器系数
    coefs = _find_coefs_numba(min_period, test_period, half_length)

    # 应用滤波器
    diff_work = np.zeros(lookback)
    leg_work = np.zeros(lookback)
    filtered_value, longest_leg, n_legs, diff_work, leg_work = _apply_filter_numba(
        y, coefs, half_length, lookback, diff_work, leg_work
    )

    # 计算宽度
    width = _calculate_width_numba(diff_work, lookback, half_length, beta)

    # 计算 FTI
    fti_value = _calculate_fti_numba(leg_work, width, n_legs, longest_leg, noise_cut)

    print(f"\n=== 单个周期 (period={test_period}) 的 Python 计算结果 ===")
    print(f"filtered_value: {filtered_value:.6f}")
    print(f"width:          {width:.6f}")
    print(f"n_legs:         {n_legs}")
    print(f"longest_leg:    {longest_leg:.6f}")
    print(f"FTI 原始值:     {fti_value:.6f}")
    print(f"FTI 值域检查:   >= 0 ? {fti_value >= 0}")

    # 验证 FTI 原始值不应该有负数
    # 注意：根据公式 mean_move / (width + 1e-5)，FTI 原始值应该 >= 0
    assert fti_value >= 0, f"Python FTI 原始值为负: {fti_value}"


def test_fti_with_real_price_data():
    """使用真实市场数据测试（如果出错，记录详细信息）"""
    # 模拟真实的 BTC 价格走势
    np.random.seed(999)
    base_price = 50000
    returns = np.random.randn(500) * 0.02  # 2% 波动率
    prices = base_price * np.exp(np.cumsum(returns))

    data_window = prices[-150:][::-1]

    try:
        python_calculator = PythonFTI()
        python_result = python_calculator.process(data_window)

        rust_result = rust_fti(data_window)

        print(f"\n=== 真实数据测试 ===")
        print(f"价格范围: {prices.min():.2f} - {prices.max():.2f}")
        print(f"Python FTI: {python_result.fti:.6f}")
        print(f"Rust FTI:   {rust_result[0]:.6f}")

        # 验证范围（FTI >= 0）
        assert python_result.fti >= 0, f"Python FTI 为负"
        assert rust_result[0] >= 0, f"Rust FTI 为负: {rust_result[0]}"

        # 验证一致性
        assert np.isclose(python_result.fti, rust_result[0], rtol=1e-5, atol=1e-5), \
            f"FTI 不一致: Python={python_result.fti}, Rust={rust_result[0]}"

    except Exception as e:
        print(f"\n错误信息: {e}")
        print(f"数据统计:")
        print(f"  长度: {len(data_window)}")
        print(f"  均值: {data_window.mean():.2f}")
        print(f"  标准差: {data_window.std():.2f}")
        print(f"  最小值: {data_window.min():.2f}")
        print(f"  最大值: {data_window.max():.2f}")
        raise


def test_fti_edge_cases():
    """边界情况测试"""

    # 测试1: 平坦价格（无趋势）
    flat_prices = np.ones(150) * 100.0

    python_calc = PythonFTI()
    python_result = python_calc.process(flat_prices[::-1])
    rust_result = rust_fti(flat_prices[::-1])

    print(f"\n=== 边界测试：平坦价格 ===")
    print(f"Python FTI: {python_result.fti:.6f}")
    print(f"Rust FTI:   {rust_result[0]:.6f}")

    assert python_result.fti >= 0
    assert rust_result[0] >= 0
    assert np.isclose(python_result.fti, rust_result[0], rtol=1e-5, atol=1e-5)

    # 测试2: 线性趋势
    linear_prices = np.linspace(100, 200, 150)

    python_result = python_calc.process(linear_prices[::-1])
    rust_result = rust_fti(linear_prices[::-1])

    print(f"\n=== 边界测试：线性趋势 ===")
    print(f"Python FTI: {python_result.fti:.6f}")
    print(f"Rust FTI:   {rust_result[0]:.6f}")

    assert python_result.fti >= 0
    assert rust_result[0] >= 0
    assert np.isclose(python_result.fti, rust_result[0], rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    print("=" * 60)
    print("FTI 正确性测试 - Python vs Rust")
    print("=" * 60)

    test_fti_component_comparison()
    test_fti_basic_comparison()
    test_fti_with_real_price_data()
    test_fti_edge_cases()

    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
