"""
共享 pytest fixtures: 从 Jesse 加载真实 BTCUSDT 1m K线数据

运行方式: 从项目根目录执行
    pytest tests/test_backtest_vectorized/
"""

from pathlib import Path

import numpy as np
import pytest

# 初始化 Jesse 数据库连接 (必须在导入 research 之前)
from jesse.services import db  # noqa: F401


_CACHE_DIR = Path("data/test_candles")
_CACHE_NPZ = (
    _CACHE_DIR / "jesse_Binance_Perpetual_Futures_BTC-USDT_1m_2025-05-01_2025-07-01.npz"
)
_CACHE_NPY = (
    _CACHE_DIR / "jesse_Binance_Perpetual_Futures_BTC-USDT_1m_2025-05-01_2025-07-01.npy"
)


def _load_cached_candles() -> np.ndarray | None:
    if _CACHE_NPZ.exists():
        data = np.load(_CACHE_NPZ, allow_pickle=False)
        if "all_candles" in data:
            return data["all_candles"]
        if "trading_candles" in data:
            return data["trading_candles"]
        raise ValueError(
            f"Invalid cache file: {_CACHE_NPZ} missing all_candles/trading_candles"
        )

    if _CACHE_NPY.exists():
        return np.load(_CACHE_NPY, allow_pickle=False)

    return None


@pytest.fixture(scope="module")
def jesse_candles():
    """
    加载真实 BTCUSDT 1m K线数据

    Returns:
        tuple[np.ndarray, np.ndarray]: (warmup_candles, trading_candles)
            - warmup_candles: 用于模型预热的 K 线数据
            - trading_candles: 用于回测交易的 K 线数据

    数据格式: NumPy array [N, 6]
        - [:, 0]: timestamp (毫秒)
        - [:, 1]: open
        - [:, 2]: close
        - [:, 3]: high
        - [:, 4]: low
        - [:, 5]: volume
    """
    from jesse import helpers, research

    # 优先使用本地缓存，避免依赖 DB 环境
    all_candles = _load_cached_candles()

    if all_candles is None:
        # 加载 2 个月数据用于测试 (2025-05-01 ~ 2025-07-01)
        # 注:
        # 1. fracdiff 特征需要较长的 lookback period
        # 2. 某些特征需要 512 窗口，需要确保 warmup fusion bars > 512
        # 3. Fusion bar 压缩比约 72:1，需要 ~40000 个 warmup candles
        try:
            _, all_candles = research.get_candles(
                "Binance Perpetual Futures",
                "BTC-USDT",
                "1m",
                helpers.date_to_timestamp("2025-05-01"),
                helpers.date_to_timestamp("2025-07-01"),
                warmup_candles_num=0,
                caching=False,
                is_for_jesse=False,
            )
        except Exception as exc:
            pytest.skip(
                "Jesse DB 未就绪且本地无缓存 candles 文件。"
                f"请先运行 scripts/cache_jesse_candles.py 生成缓存。({exc})"
            )

    # 过滤零成交量 K 线
    all_candles = all_candles[all_candles[:, 5] >= 0]
    if len(all_candles) == 0:
        pytest.skip("Candles 数据为空，跳过测试。")

    # 手动分割: 前 40000 根作为 warmup，其余作为 trading
    # 这样可以确保 warmup fusion bars > 512 (40000 / 72 ≈ 555)
    warmup_candles = all_candles[:40000]
    trading_candles = all_candles[40000:]

    print(f"\n[fixture] Warmup candles: {len(warmup_candles)}")
    print(f"[fixture] Trading candles: {len(trading_candles)}")

    return warmup_candles, trading_candles


@pytest.fixture(scope="module")
def small_candles(jesse_candles):
    """
    返回少量 K 线用于快速测试

    Returns:
        np.ndarray: 前 1000 根 warmup candles
    """
    warmup_candles, _ = jesse_candles
    return warmup_candles[:1000]


@pytest.fixture(scope="module")
def fusion_bar_instance():
    """
    返回一个新的 DemoBar 实例

    Returns:
        DemoBar: 配置好的 fusion bar 生成器
    """
    from src.bars.fusion.demo import DemoBar

    return DemoBar(clip_r=0.012, max_bars=-1, threshold=1.399)
