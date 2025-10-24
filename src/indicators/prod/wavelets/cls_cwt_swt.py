import numpy as np
import pywt
from jesse.helpers import get_candle_source

from pyrs_indicators.ind_wavelets import cwt

from src.indicators.prod._indicator_base._cls_ind import IndicatorBase

SAMPLING_HOURS = 0.5
MIN_SCALE = 8
MAX_SCALE = 128
NUM_SCALES = 64
DYNAMIC_RANGE = (5, 95)

_fs = 1 / SAMPLING_HOURS
_scales = np.logspace(
    np.log2(MIN_SCALE),
    np.log2(MAX_SCALE),
    num=NUM_SCALES,
    base=2,
)
_scales_freq = pywt.scale2frequency("cmor1.5-1.0", _scales) * _fs
_valid_scales = _scales[(_scales_freq > 0.1) & (_scales_freq < _fs / 2)]
_pad_width = round(max(_valid_scales))


def _cwt(src: np.ndarray):
    """
    Compute CWT using Rust implementation (3.5x faster than PyWavelets).

    Rust implementation provides significant speedup with perfect numerical
    alignment (error < 3e-14).
    """
    if len(_valid_scales) < 5:
        raise ValueError("Not enough valid scales")

    # 使用新的 Python 接口
    cwt_dB, _freqs = cwt(
        src,
        _valid_scales,
        wavelet="cmor1.5-1.0",
        sampling_period=SAMPLING_HOURS,
        precision=12,
        pad_width=_pad_width,
        verbose=False,  # Production mode: no progress output
    )
    return cwt_dB


class CWT_SWT(IndicatorBase):
    def __init__(
        self,
        candles: np.ndarray,
        window: int,
        source_type: str = "close",
        sequential: bool = False,
    ):
        super().__init__(candles, sequential)
        self.window = window
        self.src = get_candle_source(candles, source_type)

        self.process()

    def _single_process(self):
        single_res = _cwt(self.src[-self.window :])
        self.raw_result.append(single_res)

    def _sequential_process(self):
        src_with_window = [
            self.src[idx - self.window : idx]
            for idx in range(self.window, len(self.src) + 1)
        ]
        # Rust implementation has internal parallelization via rayon
        # Direct loop is sufficient (no need for joblib)
        res = [_cwt(i) for i in src_with_window]

        self.raw_result.extend(res)
