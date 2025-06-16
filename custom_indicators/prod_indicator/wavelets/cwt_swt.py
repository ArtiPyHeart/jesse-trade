import numpy as np
import pywt
from jesse.helpers import get_candle_source, slice_candles

SAMPLING_HOURS = 0.5
MIN_SCALE = 8
MAX_SCALE = 128
NUM_SCALES = 64
DYNAMIC_RANGE = (5, 95)


def cwt(candles: np.ndarray, source_type: str = "close", sequential: bool = False):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    fs = 1 / SAMPLING_HOURS
    scales = np.logspace(
        np.log2(MIN_SCALE),
        np.log2(MAX_SCALE),
        num=NUM_SCALES,
        base=2,
    )

    scales_freq = pywt.scale2frequency("cmor1.5-1.0", scales) * fs
    valid_scales = scales[(scales_freq > 0.1) & (scales_freq < fs / 2)]
    if len(valid_scales) < 5:
        raise ValueError("Not enough valid scales")

    cwtmat, _freqs = pywt.cwt(
        src,
        valid_scales,
        "cmor1.5-1.0",
        sampling_period=SAMPLING_HOURS,
    )

    cwt_dB = np.log10(np.abs(cwtmat) + 1e-12).T
    if sequential:
        return cwt_dB
    else:
        return cwt_dB[-1]
