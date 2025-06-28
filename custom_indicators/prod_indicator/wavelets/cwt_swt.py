import numpy as np
import pywt
from jesse.helpers import get_candle_source, slice_candles
from mpire.pool import WorkerPool

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
    return cwt_dB[-1].tolist()


def cwt(
    candles: np.ndarray,
    window: int,
    parallel: bool = False,
    source_type: str = "close",
    sequential: bool = False,
):
    candles = slice_candles(candles, sequential)
    src = get_candle_source(candles, source_type)

    if sequential:
        if parallel:
            res = [src[idx - window : idx] for idx in range(window, len(src))]
            with WorkerPool() as pool:
                res = pool.map(_cwt, res)
        else:
            res = []
            for idx in range(window, len(src)):
                res.append(_cwt(src[idx - window : idx]))
        res = np.asarray(res)
        columns = res.shape[1]
        # padding res with nan (window, columns)
        res = np.vstack([np.full((window, columns), np.nan), res])
        return res
    else:
        return np.asarray(_cwt(src[-window:]))
