import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.plot.plot_kde import process_sign_sequence


def test_process_sign_sequence_shift_next():
    values = np.array([1.0, -2.0, -3.0, 4.0])
    expected = np.array([-1.0, 2.0, -3.0, 4.0])
    result = process_sign_sequence(values, compare_shift=-1)
    assert np.allclose(result, expected)


def test_process_sign_sequence_shift_prev():
    values = np.array([1.0, -2.0, -3.0, 4.0])
    expected = np.array([1.0, -2.0, 3.0, -4.0])
    result = process_sign_sequence(values, compare_shift=1)
    assert np.allclose(result, expected)


def test_process_sign_sequence_invalid_shift():
    values = np.array([1.0, 2.0])
    with pytest.raises(AssertionError):
        process_sign_sequence(values, compare_shift=0)


def test_process_sign_sequence_short_values():
    values = np.array([2.0])
    result = process_sign_sequence(values, compare_shift=-1)
    assert np.allclose(result, values)
