"""
多窗口熵特征集合
使用函数工厂模式批量生成特征
"""

import numpy as np

from src.indicators.prod import sample_entropy_indicator, approximate_entropy_indicator
from ..registry import feature


# 批量注册多窗口熵特征 - 简化版
windows = [32, 64, 128, 256, 512]

for w in windows:
    # Sample Entropy - Spot
    # 使用默认参数 _w=w 直接捕获当前窗口值，避免闭包晚绑定问题
    @feature(
        name=f"sample_entropy_w{w}_spot",
        params={"period": w, "use_array_price": False},
        description=f"Sample Entropy (window={w}, spot price)",
    )
    def sample_entropy_spot(
        candles: np.ndarray,
        sequential: bool = True,
        period: int = None,
        use_array_price: bool = False,
        _w=w,  # 关键：用默认参数捕获当前循环的窗口值
    ):
        return sample_entropy_indicator(
            candles,
            period=period or _w,
            use_array_price=use_array_price,
            sequential=sequential,
        )

    # Sample Entropy - Array
    @feature(
        name=f"sample_entropy_w{w}_array",
        params={"period": w, "use_array_price": True},
        description=f"Sample Entropy (window={w}, array price)",
    )
    def sample_entropy_array(
        candles: np.ndarray,
        sequential: bool = True,
        period: int = None,
        use_array_price: bool = True,
        _w=w,
    ):
        return sample_entropy_indicator(
            candles,
            period=period or _w,
            use_array_price=use_array_price,
            sequential=sequential,
        )

    # Approximate Entropy - Spot
    @feature(
        name=f"approximate_entropy_w{w}_spot",
        params={"period": w, "use_array_price": False},
        description=f"Approximate Entropy (window={w}, spot price)",
    )
    def approximate_entropy_spot(
        candles: np.ndarray,
        sequential: bool = True,
        period: int = None,
        use_array_price: bool = False,
        _w=w,
    ):
        return approximate_entropy_indicator(
            candles,
            period=period or _w,
            use_array_price=use_array_price,
            sequential=sequential,
        )

    # Approximate Entropy - Array
    @feature(
        name=f"approximate_entropy_w{w}_array",
        params={"period": w, "use_array_price": True},
        description=f"Approximate Entropy (window={w}, array price)",
    )
    def approximate_entropy_array(
        candles: np.ndarray,
        sequential: bool = True,
        period: int = None,
        use_array_price: bool = True,
        _w=w,
    ):
        return approximate_entropy_indicator(
            candles,
            period=period or _w,
            use_array_price=use_array_price,
            sequential=sequential,
        )
