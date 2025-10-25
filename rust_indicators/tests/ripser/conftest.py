"""
Pytest 配置和共享 fixtures

提供所有测试模块共享的工具和数据。
"""

import numpy as np
import pytest
from pathlib import Path

# 参考数据目录
REFERENCE_DATA_DIR = Path(__file__).parent / "reference_data"


@pytest.fixture
def reference_data_dir():
    """参考数据目录 fixture"""
    REFERENCE_DATA_DIR.mkdir(exist_ok=True)
    return REFERENCE_DATA_DIR


@pytest.fixture
def simple_point_cloud():
    """简单点云：圆形上的 10 个点"""
    n_points = 10
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return points


@pytest.fixture
def noisy_circle():
    """带噪声的圆形点云"""
    n_points = 50
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radius = 1.0 + np.random.normal(0, 0.1, n_points)
    points = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta)
    ])
    return points


@pytest.fixture
def time_series_1d():
    """1D 时间序列（模拟你的使用场景）"""
    # 模拟价格数据：趋势 + 周期性 + 噪声
    t = np.linspace(0, 10, 100)
    trend = 0.5 * t
    seasonal = 2 * np.sin(2 * np.pi * t)
    noise = np.random.normal(0, 0.2, len(t))
    signal = trend + seasonal + noise
    return signal


def assert_arrays_close(actual, expected, rtol=1e-6, atol=1e-6, name="array"):
    """
    断言两个数组近似相等

    Args:
        actual: 实际值
        expected: 期望值
        rtol: 相对容差
        atol: 绝对容差
        name: 数组名称（用于错误信息）
    """
    np.testing.assert_allclose(
        actual, expected,
        rtol=rtol, atol=atol,
        err_msg=f"{name} mismatch"
    )


def compute_max_error(actual, expected):
    """
    计算最大误差

    Returns:
        float: 最大绝对误差
    """
    return np.max(np.abs(actual - expected))
