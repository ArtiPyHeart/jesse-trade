# PyTorch CPU configuration will be triggered when importing from research or src modules
# Import from research first to ensure PyTorch is configured before other imports
from research.plot import analyze_confidence_slices

import lightgbm as lgb
import numpy as np
import pandas as pd
from jesse.helpers import date_to_timestamp
from strategies.BinanceBtcDeapV1Voting.models.config import (
    FEAT_L6,
    FEAT_FRACDIFF,
    DeepSSMContainer,
    LGSSMContainer,
)


def main():
    try:

        print("Step 1: 初始化模型容器...")
        model_deep_ssm = DeepSSMContainer()
        model_lg_ssm = LGSSMContainer()
        print("✓ 模型容器初始化完成")

        print("Step 2: 加载特征数据...")
        df_feat_raw = pd.read_parquet("data/feat_hard_L4.parquet")
        print(f"✓ 加载特征数据: shape={df_feat_raw.shape}")

        print("Step 3: 转换特征数据...")
        print(f"  - FEAT_FRACDIFF columns: {len(FEAT_FRACDIFF)}")
        df_deep_ssm = model_deep_ssm.transform(df_feat_raw[FEAT_FRACDIFF])
        print(f"  - df_deep_ssm shape: {df_deep_ssm.shape}")

        df_lg_ssm = model_lg_ssm.transform(df_feat_raw[FEAT_FRACDIFF])
        print(f"  - df_lg_ssm shape: {df_lg_ssm.shape}")

        df_feat_l4_full = pd.concat([df_deep_ssm, df_lg_ssm, df_feat_raw], axis=1)[
            FEAT_L6
        ]
        print(f"✓ 特征转换完成: shape={df_feat_l4_full.shape}")

        print("Step 4: 筛选测试数据时间范围...")
        test_features = df_feat_l4_full[
            (df_feat_l4_full.index.to_numpy() >= date_to_timestamp("2025-03-01"))
            & (df_feat_l4_full.index.to_numpy() < date_to_timestamp("2025-06-30"))
        ]
        print(f"✓ 测试特征: shape={test_features.shape}")

        print("Step 5: 加载K线数据...")
        candles = np.load("data/bar_deap_v1.npy")
        print(f"  - K线数据 shape: {candles.shape}")
        timestamps = candles[:, 0].astype(np.int64)
        mask = (timestamps >= test_features.index[0]) & (
            timestamps <= test_features.index[-1]
        )

        volume = candles[:, 5][mask]
        print(f"  - Volume数据长度: {len(volume)}")

        datetimes = pd.to_datetime(test_features.index, unit="ms")
        print(f"  - 时间数据长度: {len(datetimes)}")

        assert len(volume) == len(
            datetimes
        ), f"数据长度不匹配: volume={len(volume)}, datetimes={len(datetimes)}"
        print("✓ K线数据处理完成")

        print("Step 6: 加载LightGBM模型...")
        model_l4 = lgb.Booster(
            model_file="strategies/BinanceBtcDeapV1Voting/models/model_l6.txt"
        )
        print("✓ 模型加载成功")

        print("Step 7: 预测...")
        res = model_l4.predict(test_features)
        print(
            f"✓ 预测完成: 结果长度={len(res)}, min={np.min(res):.4f}, max={np.max(res):.4f}"
        )

        assert len(res) == len(
            datetimes
        ), f"预测结果长度不匹配: res={len(res)}, datetimes={len(datetimes)}"

        print("Step 8: 执行置信度切片分析...")
        analyze_confidence_slices(
            datetimes,
            res,
            volume,
            full_range=True,
            granularity=0.01,
        )
        print("✓ 分析完成")

    except Exception as e:
        print(f"\n❌ 错误发生在: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
