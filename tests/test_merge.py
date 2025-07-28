import numpy as np
import pytest

from bar import np_merge_bars

# --------------------------- 辅助函数 ---------------------------


def _random_candles(n: int, seed: int = 42) -> np.ndarray:
    """生成随机 K 线数据 (timestamp, open, close, high, low, volume)。"""
    rng = np.random.default_rng(seed)

    t = np.arange(n, dtype=np.float64)
    o = rng.random(n) * 100.0  # 开盘价 0~100
    c = o + rng.normal(0.0, 1.0, n)  # 收盘价围绕开盘价波动

    # 保证 high ≥ open,close 且 low ≤ open,close
    h = np.maximum(o, c) + rng.random(n)
    l = np.minimum(o, c) - rng.random(n)

    v = rng.integers(100, 1_000, n, dtype=np.int64).astype(np.float64)  # 成交量

    return np.column_stack((t, o, c, h, l, v))


# --------------------------- 测试用例 ---------------------------


@pytest.mark.parametrize(
    "n,bars_limit,lag",
    [
        (50, 20, 1),
        (200, 80, 1),
        (1000, 100, 2),
    ],
)
def test_np_merge_bars_consistency(n: int, bars_limit: int, lag: int) -> None:
    """验证 use_fast=True 与 use_fast=False 的结果应完全一致。"""
    candles = _random_candles(n)

    fast_res = np_merge_bars(candles.copy(), bars_limit, lag=lag, use_fast=True)
    slow_res = np_merge_bars(candles.copy(), bars_limit, lag=lag, use_fast=False)

    # 形状必须一致
    assert fast_res.shape == slow_res.shape == (bars_limit, 6)

    # 数值一致 (浮点允许极小误差)
    np.testing.assert_allclose(fast_res, slow_res, rtol=1e-12, atol=1e-12)
