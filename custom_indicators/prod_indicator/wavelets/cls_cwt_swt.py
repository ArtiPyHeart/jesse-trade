import numpy as np
import pywt
from joblib import Parallel, delayed
from jesse.helpers import get_candle_source

from custom_indicators.prod_indicator._indicator_base._cls_ind import IndicatorBase

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
    if len(_valid_scales) < 5:
        raise ValueError("Not enough valid scales")

    x_pad = pywt.pad(src, _pad_width, mode="symmetric")
    cwtmat, _freqs = pywt.cwt(
        x_pad,
        _valid_scales,
        "cmor1.5-1.0",
        sampling_period=SAMPLING_HOURS,
    )
    cwtmat = cwtmat[:, _pad_width:-_pad_width]
    cwt_dB = np.log10(np.abs(cwtmat) + 1e-12).T
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
            for idx in range(self.window, len(self.src))
        ]
        res = Parallel(n_jobs=-2)(delayed(_cwt)(i) for i in src_with_window)

        self.raw_result.extend(res)
