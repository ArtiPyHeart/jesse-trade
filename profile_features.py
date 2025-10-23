"""
特征计算性能分析脚本

功能：
1. 使用 research.get_candles 获取约10个月的K线数据，拆分为两段：
   - 前5个月：预热数据（触发 Numba JIT 编译）
   - 后5个月：测试数据（实际性能测试）
2. 用 DemoBar 加工为 fusion candles
3. 使用 SimpleFeatureCalculator 逐个测试所有 BUILDIN_FEATURES
4. 每个特征预热后多次运行取平均值和标准差
5. 输出 CSV 报告，按耗时降序排列
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from jesse import helpers, research
from src.bars.fusion.demo import DemoBar
from src.features.simple_feature_calculator import SimpleFeatureCalculator
from src.features.simple_feature_calculator.buildin.feature_names import BUILDIN_FEATURES


def profile_features(
    warmup_bars: np.ndarray,
    test_bars: np.ndarray,
    num_runs: int = 5,
    output_path: Path = Path("feature_profile_results.csv"),
) -> pd.DataFrame:
    """
    分析所有内置特征的计算性能

    Args:
        warmup_bars: 预热用的 Fusion Bars（触发 JIT 编译）
        test_bars: 测试用的 Fusion Bars（实际性能测试）
        num_runs: 每个特征运行次数（用于计算平均值）
        output_path: 输出CSV路径

    Returns:
        性能分析 DataFrame
    """
    print("=" * 60)
    print("特征性能分析开始")
    print("=" * 60)
    print(f"预热 Fusion Bars 数量: {len(warmup_bars):,}")
    print(f"测试 Fusion Bars 数量: {len(test_bars):,}")
    print(f"待测试特征数量: {len(BUILDIN_FEATURES)}")
    print(f"每特征运行次数: {num_runs}")
    print("=" * 60 + "\n")

    if len(warmup_bars) == 0 or len(test_bars) == 0:
        raise ValueError("Warmup 或 Test Fusion Bars 为空，请检查输入数据")

    # ========== Step 1: 逐个测试特征 ==========
    print("Step 1: 逐个测试特征性能...")
    results = []

    fc = SimpleFeatureCalculator()

    for feature_name in tqdm(BUILDIN_FEATURES, desc="测试特征", unit="feat"):
        # ========== Warmup Phase ==========
        # 使用 warmup_bars 预热，触发 numba JIT 编译
        fc.load(warmup_bars, sequential=True)
        _ = fc.get([feature_name])

        # ========== Timed Phase ==========
        # 使用 test_bars 进行多次测试
        elapsed_times = []
        feature_data = None

        for run_idx in range(num_runs):
            fc.load(test_bars, sequential=True)

            start_time = time.perf_counter()
            feature_data = fc.get([feature_name])
            end_time = time.perf_counter()

            elapsed_times.append(end_time - start_time)

        # 计算统计量
        elapsed_time_mean = np.mean(elapsed_times)
        elapsed_time_std = np.std(elapsed_times)
        elapsed_time_min = np.min(elapsed_times)
        elapsed_time_max = np.max(elapsed_times)

        # 检查输出（使用最后一次运行的结果）
        if feature_name not in feature_data:
            status = "missing"
            output_shape = "N/A"
            output_len = 0
        else:
            values = feature_data[feature_name]
            status = "success"
            if isinstance(values, np.ndarray):
                output_shape = str(values.shape)
                output_len = len(values)
            elif isinstance(values, (list, tuple)):
                output_shape = f"({len(values)},)"
                output_len = len(values)
            else:
                output_shape = "scalar"
                output_len = 1

        results.append(
            {
                "feature_name": feature_name,
                "elapsed_time_sec_mean": elapsed_time_mean,
                "elapsed_time_sec_std": elapsed_time_std,
                "elapsed_time_sec_min": elapsed_time_min,
                "elapsed_time_sec_max": elapsed_time_max,
                "elapsed_time_ms_mean": elapsed_time_mean * 1000,
                "elapsed_time_ms_std": elapsed_time_std * 1000,
                "status": status,
                "output_shape": output_shape,
                "output_len": output_len,
                "num_runs": num_runs,
            }
        )

    # ========== Step 2: 生成报告 ==========
    print("\n" + "=" * 60)
    print("Step 2: 生成性能报告")
    print("=" * 60)

    df_results = pd.DataFrame(results)

    # 按平均耗时降序排列
    df_results = df_results.sort_values(
        "elapsed_time_sec_mean", ascending=False
    ).reset_index(drop=True)

    # 添加排名
    df_results.insert(0, "rank", range(1, len(df_results) + 1))

    # 计算统计信息
    total_time = df_results["elapsed_time_sec_mean"].sum()
    success_count = (df_results["status"] == "success").sum()
    error_count = (df_results["status"] == "error").sum()
    missing_count = (df_results["status"] == "missing").sum()

    print(f"\n总测试特征数: {len(df_results)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"缺失: {missing_count}")
    print(f"总耗时（平均值之和）: {total_time:.2f} 秒")
    print(f"平均耗时: {total_time / len(df_results):.4f} 秒")

    # 显示最慢的10个特征
    print("\n最慢的10个特征:")
    print("-" * 80)
    print(
        f"{'Rank':<5} {'Feature':<40} {'Mean(ms)':<12} {'Std(ms)':<12} {'Status':<10}"
    )
    print("-" * 80)
    top10 = df_results.head(10)
    for _, row in top10.iterrows():
        print(
            f"{row['rank']:<5} {row['feature_name']:<40} "
            f"{row['elapsed_time_ms_mean']:<12.2f} "
            f"{row['elapsed_time_ms_std']:<12.2f} "
            f"{row['status']:<10}"
        )

    # 保存到CSV
    df_results.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    print("=" * 60 + "\n")

    return df_results


# ==================== 主入口 ====================
if __name__ == "__main__":
    # ========== 配置 ==========
    # 数据范围：约10个月，拆分为两段
    # Warmup: 2025-01-01 ~ 2025-06-01 (5个月)
    # Test:   2025-06-01 ~ 2025-11-01 (5个月)
    WARMUP_START = "2024-01-01"
    WARMUP_END = "2024-06-01"
    TEST_START = "2024-06-01"
    TEST_END = "2024-11-01"

    # 测试参数
    NUM_RUNS = 5  # 每个特征运行次数
    OUTPUT_CSV = Path("feature_profile_results.csv")

    # ========== 获取预热数据 ==========
    print("=" * 60)
    print("正在加载预热K线数据...")
    print("=" * 60)
    _, warmup_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(WARMUP_START),
        helpers.date_to_timestamp(WARMUP_END),
        warmup_candles_num=0,
        caching=True,
        is_for_jesse=False,
    )
    warmup_candles = warmup_candles[warmup_candles[:, 5] >= 0]
    print(f"预热数据: {len(warmup_candles):,} 根K线")
    print(f"时间范围: {WARMUP_START} ~ {WARMUP_END}\n")

    # ========== 获取测试数据 ==========
    print("正在加载测试K线数据...")
    _, test_candles = research.get_candles(
        "Binance Perpetual Futures",
        "BTC-USDT",
        "1m",
        helpers.date_to_timestamp(TEST_START),
        helpers.date_to_timestamp(TEST_END),
        warmup_candles_num=0,
        caching=True,
        is_for_jesse=False,
    )
    test_candles = test_candles[test_candles[:, 5] >= 0]
    print(f"测试数据: {len(test_candles):,} 根K线")
    print(f"时间范围: {TEST_START} ~ {TEST_END}\n")

    # ========== 生成 Fusion Bars ==========
    print("=" * 60)
    print("生成预热 Fusion Bars...")
    print("=" * 60)
    warmup_bar_container = DemoBar(max_bars=20000)
    warmup_bar_container.update_with_candles(warmup_candles)
    warmup_bars = warmup_bar_container.get_fusion_bars()
    print(f"预热 Fusion Bars: {len(warmup_bars):,} 个\n")

    print("生成测试 Fusion Bars...")
    test_bar_container = DemoBar(max_bars=20000)
    test_bar_container.update_with_candles(test_candles)
    test_bars = test_bar_container.get_fusion_bars()
    print(f"测试 Fusion Bars: {len(test_bars):,} 个\n")

    # ========== 运行性能分析 ==========
    df_profile = profile_features(
        warmup_bars=warmup_bars,
        test_bars=test_bars,
        num_runs=NUM_RUNS,
        output_path=OUTPUT_CSV,
    )

    print("性能分析完成！")
