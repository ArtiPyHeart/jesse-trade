"""ADF 基准测试（statsmodels vs arch）

说明：
- 使用相同的窗口样本和 autolag=AIC 设置进行对比。
- 仅用于手动运行的基准测试，不对速度做硬性断言。
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pytest

try:
    from arch.unitroot import ADF
except Exception:  # pragma: no cover - optional dependency
    ADF = None

from statsmodels.tsa.stattools import adfuller

DATA_PATH = Path("data/btc_1m.npy")


def _load_close() -> np.ndarray:
    if not DATA_PATH.exists():
        pytest.skip(f"数据文件不存在: {DATA_PATH}")
    candles = np.load(DATA_PATH)
    if candles.ndim != 2 or candles.shape[1] != 6:
        pytest.skip("candles 格式不符合 Jesse 6 列规范")
    return candles[:, 2].astype(float)


def _sample_windows(
    close: np.ndarray,
    window_size: int,
    n_samples: int,
    seed: int = 7,
) -> list[np.ndarray]:
    max_start = len(close) - window_size
    if max_start <= 0:
        pytest.skip("close 序列长度不足")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=n_samples)
    return [close[i : i + window_size] for i in starts]


def _time_statsmodels(windows: list[np.ndarray]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = adfuller(windows[0], autolag="AIC", regression="c")  # warmup
        t0 = time.perf_counter()
        for w in windows:
            _ = adfuller(w, autolag="AIC", regression="c")
        return time.perf_counter() - t0


def _time_arch(windows: list[np.ndarray]) -> float:
    if ADF is None:
        pytest.skip("arch 未安装或导入失败")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = ADF(windows[0], trend="c", method="aic")  # warmup
        t0 = time.perf_counter()
        for w in windows:
            _ = ADF(w, trend="c", method="aic")
        return time.perf_counter() - t0


def test_adf_benchmark() -> None:
    close = _load_close()

    # 基准配置：窗口与样本数可按需调整
    window_size = 120
    n_samples = 50

    windows = _sample_windows(close, window_size, n_samples)

    t_sm = _time_statsmodels(windows)
    t_arch = _time_arch(windows)

    print(
        "\nADF benchmark (autolag=AIC, regression/trend=c)\n"
        f"  statsmodels: {t_sm:.4f}s for {n_samples} windows\n"
        f"  arch:        {t_arch:.4f}s for {n_samples} windows\n"
        f"  speedup:     {t_sm / t_arch:.2f}x (statsmodels/arch)"
    )

    assert t_sm > 0.0
    assert t_arch > 0.0
