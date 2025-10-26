import numpy as np

from src.features.simple_feature_calculator import (
    class_feature,
)
from src.indicators.prod import VMD_NRBO
from src.indicators.prod.wavelets import CWT_SWT_Rust as CWT_SWT


# 注册类型特征


# 基础特征类
class BaseIndicatorFeature:
    """指标特征基类"""

    def __init__(
        self,
        indicator_class,
        candles: np.ndarray,
        window: int,
        sequential: bool = False,
        **kwargs
    ):
        self.indicator = indicator_class(candles, window, sequential=sequential)

    @property
    def raw_result(self):
        """暴露indicator的raw_result供转换链使用"""
        return self.indicator.raw_result

    def res(self, **kwargs):
        return self.indicator.res(**kwargs)


# 工厂函数
def create_indicator_feature_class(indicator_class, class_name):
    """创建指标特征类的工厂函数"""

    class DynamicFeature(BaseIndicatorFeature):
        def __init__(
            self, candles: np.ndarray, window: int, sequential: bool = False, **kwargs
        ):
            super().__init__(indicator_class, candles, window, sequential, **kwargs)

    DynamicFeature.__name__ = class_name
    DynamicFeature.__qualname__ = class_name

    return DynamicFeature


# 创建CWT特征类
CWTFeature32 = create_indicator_feature_class(CWT_SWT, "CWTFeature32")
CWTFeature64 = create_indicator_feature_class(CWT_SWT, "CWTFeature64")
CWTFeature128 = create_indicator_feature_class(CWT_SWT, "CWTFeature128")
CWTFeature256 = create_indicator_feature_class(CWT_SWT, "CWTFeature256")
CWTFeature512 = create_indicator_feature_class(CWT_SWT, "CWTFeature512")

# 创建VMD特征类
VMDFeature32 = create_indicator_feature_class(VMD_NRBO, "VMDFeature32")
VMDFeature64 = create_indicator_feature_class(VMD_NRBO, "VMDFeature64")
VMDFeature128 = create_indicator_feature_class(VMD_NRBO, "VMDFeature128")
VMDFeature256 = create_indicator_feature_class(VMD_NRBO, "VMDFeature256")
VMDFeature512 = create_indicator_feature_class(VMD_NRBO, "VMDFeature512")


# 注册CWT特征
@class_feature(
    name="cwt_w32",
    params={"window": 32},
    returns_multiple=True,
    description="CWT SWT Indicator",
)
class _CWTFeature32(CWTFeature32):
    """CWT SWT特征 - Window 32"""

    def __init__(
        self, candles: np.ndarray, window: int = 32, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="cwt_w64",
    params={"window": 64},
    returns_multiple=True,
    description="CWT SWT Indicator",
)
class _CWTFeature64(CWTFeature64):
    """CWT SWT特征 - Window 64"""

    def __init__(
        self, candles: np.ndarray, window: int = 64, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="cwt_w128",
    params={"window": 128},
    returns_multiple=True,
    description="CWT SWT Indicator",
)
class _CWTFeature128(CWTFeature128):
    """CWT SWT特征 - Window 128"""

    def __init__(
        self, candles: np.ndarray, window: int = 128, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="cwt_w256",
    params={"window": 256},
    returns_multiple=True,
    description="CWT SWT Indicator",
)
class _CWTFeature256(CWTFeature256):
    """CWT SWT特征 - Window 256"""

    def __init__(
        self, candles: np.ndarray, window: int = 256, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="cwt_w512",
    params={"window": 512},
    returns_multiple=True,
    description="CWT SWT Indicator",
)
class _CWTFeature512(CWTFeature512):
    """CWT SWT特征 - Window 512"""

    def __init__(
        self, candles: np.ndarray, window: int = 512, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


# 注册VMD特征
@class_feature(
    name="vmd_w32",
    params={"window": 32},
    returns_multiple=True,
    description="VMD NRBO Indicator",
)
class _VMDFeature32(VMDFeature32):
    """VMD NRBO特征 - Window 32"""

    def __init__(
        self, candles: np.ndarray, window: int = 32, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="vmd_w64",
    params={"window": 64},
    returns_multiple=True,
    description="VMD NRBO Indicator",
)
class _VMDFeature64(VMDFeature64):
    """VMD NRBO特征 - Window 64"""

    def __init__(
        self, candles: np.ndarray, window: int = 64, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="vmd_w128",
    params={"window": 128},
    returns_multiple=True,
    description="VMD NRBO Indicator",
)
class _VMDFeature128(VMDFeature128):
    """VMD NRBO特征 - Window 128"""

    def __init__(
        self, candles: np.ndarray, window: int = 128, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="vmd_w256",
    params={"window": 256},
    returns_multiple=True,
    description="VMD NRBO Indicator",
)
class _VMDFeature256(VMDFeature256):
    """VMD NRBO特征 - Window 256"""

    def __init__(
        self, candles: np.ndarray, window: int = 256, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)


@class_feature(
    name="vmd_w512",
    params={"window": 512},
    returns_multiple=True,
    description="VMD NRBO Indicator",
)
class _VMDFeature512(VMDFeature512):
    """VMD NRBO特征 - Window 512"""

    def __init__(
        self, candles: np.ndarray, window: int = 512, sequential: bool = False, **kwargs
    ):
        super().__init__(candles, window, sequential, **kwargs)
