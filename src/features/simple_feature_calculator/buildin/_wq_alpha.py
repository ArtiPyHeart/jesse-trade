"""
WorldQuant 101 Alphas 特征注册

将 WQ101 Alpha 因子注册到 SimpleFeatureCalculator
"""

import numpy as np

from src.features.simple_feature_calculator import feature
from src.indicators.prod.wq_alpha import (
    alpha_001, alpha_002, alpha_003, alpha_004, alpha_005,
    alpha_006, alpha_007, alpha_008, alpha_009, alpha_010,
    alpha_011, alpha_012, alpha_013, alpha_014, alpha_015,
    alpha_016, alpha_017, alpha_018, alpha_019, alpha_020,
    alpha_021, alpha_022, alpha_023, alpha_024, alpha_025,
    alpha_026, alpha_027, alpha_030, alpha_033, alpha_034,
    alpha_035, alpha_036, alpha_037, alpha_038, alpha_039,
    alpha_040, alpha_041, alpha_042, alpha_043, alpha_044,
    alpha_045, alpha_046, alpha_047, alpha_049, alpha_050,
    alpha_051, alpha_052, alpha_053, alpha_054, alpha_055,
    alpha_057, alpha_061, alpha_062, alpha_064, alpha_065,
    alpha_066, alpha_068, alpha_071, alpha_072, alpha_073,
    alpha_074, alpha_075, alpha_077, alpha_078, alpha_081,
    alpha_083, alpha_084, alpha_085, alpha_086, alpha_088,
    alpha_092, alpha_094, alpha_095, alpha_096, alpha_098,
    alpha_099, alpha_101,
)


def _ensure_array(result) -> np.ndarray:
    """确保返回 numpy array"""
    if not isinstance(result, np.ndarray):
        result = np.array([result])
    return result


# ============ Price/Volume Alphas ============

@feature(name="wq_alpha_001", description="WQ101 Alpha #1: Conditioned Returns Momentum")
def wq_alpha_001(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_001(candles, sequential=sequential))


@feature(name="wq_alpha_009", description="WQ101 Alpha #9: Close Momentum Conditional")
def wq_alpha_009(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_009(candles, sequential=sequential))


@feature(name="wq_alpha_012", description="WQ101 Alpha #12: Volume-Price Reversal")
def wq_alpha_012(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_012(candles, sequential=sequential))


@feature(name="wq_alpha_021", description="WQ101 Alpha #21: Volume-ADV Conditional")
def wq_alpha_021(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_021(candles, sequential=sequential))


@feature(name="wq_alpha_023", description="WQ101 Alpha #23: High Momentum Conditional")
def wq_alpha_023(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_023(candles, sequential=sequential))


@feature(name="wq_alpha_024", description="WQ101 Alpha #24: Long-term Close Momentum")
def wq_alpha_024(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_024(candles, sequential=sequential))


@feature(name="wq_alpha_046", description="WQ101 Alpha #46: Close Momentum Conditional")
def wq_alpha_046(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_046(candles, sequential=sequential))


@feature(name="wq_alpha_049", description="WQ101 Alpha #49: Momentum Threshold")
def wq_alpha_049(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_049(candles, sequential=sequential))


@feature(name="wq_alpha_051", description="WQ101 Alpha #51: Momentum Threshold 2")
def wq_alpha_051(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_051(candles, sequential=sequential))


@feature(name="wq_alpha_053", description="WQ101 Alpha #53: Price Position Delta")
def wq_alpha_053(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_053(candles, sequential=sequential))


@feature(name="wq_alpha_054", description="WQ101 Alpha #54: Price Ratio")
def wq_alpha_054(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_054(candles, sequential=sequential))


@feature(name="wq_alpha_101", description="WQ101 Alpha #101: Intraday Price Move Ratio")
def wq_alpha_101(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_101(candles, sequential=sequential))


# ============ Correlation-based Alphas ============

@feature(name="wq_alpha_002", description="WQ101 Alpha #2: Volume-Return Correlation")
def wq_alpha_002(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_002(candles, sequential=sequential))


@feature(name="wq_alpha_003", description="WQ101 Alpha #3: Open-Volume Correlation")
def wq_alpha_003(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_003(candles, sequential=sequential))


@feature(name="wq_alpha_006", description="WQ101 Alpha #6: Open-Volume Correlation 10")
def wq_alpha_006(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_006(candles, sequential=sequential))


@feature(name="wq_alpha_014", description="WQ101 Alpha #14: Returns-Open-Volume Correlation")
def wq_alpha_014(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_014(candles, sequential=sequential))


@feature(name="wq_alpha_015", description="WQ101 Alpha #15: High-Volume Correlation Sum")
def wq_alpha_015(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_015(candles, sequential=sequential))


@feature(name="wq_alpha_018", description="WQ101 Alpha #18: Intraday Move Correlation")
def wq_alpha_018(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_018(candles, sequential=sequential))


@feature(name="wq_alpha_022", description="WQ101 Alpha #22: High-Volume Correlation Delta")
def wq_alpha_022(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_022(candles, sequential=sequential))


@feature(name="wq_alpha_026", description="WQ101 Alpha #26: Volume-High Correlation Max")
def wq_alpha_026(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_026(candles, sequential=sequential))


@feature(name="wq_alpha_037", description="WQ101 Alpha #37: Correlation Intraday Move")
def wq_alpha_037(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_037(candles, sequential=sequential))


@feature(name="wq_alpha_040", description="WQ101 Alpha #40: High Volatility Correlation")
def wq_alpha_040(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_040(candles, sequential=sequential))


@feature(name="wq_alpha_044", description="WQ101 Alpha #44: High-Volume Correlation")
def wq_alpha_044(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_044(candles, sequential=sequential))


@feature(name="wq_alpha_045", description="WQ101 Alpha #45: Close-Volume Correlation")
def wq_alpha_045(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_045(candles, sequential=sequential))


@feature(name="wq_alpha_055", description="WQ101 Alpha #55: Price Position Volume Correlation")
def wq_alpha_055(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_055(candles, sequential=sequential))


@feature(name="wq_alpha_068", description="WQ101 Alpha #68: High-ADV Correlation")
def wq_alpha_068(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_068(candles, sequential=sequential))


@feature(name="wq_alpha_085", description="WQ101 Alpha #85: Correlation Power")
def wq_alpha_085(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_085(candles, sequential=sequential))


@feature(name="wq_alpha_088", description="WQ101 Alpha #88: Min Decay Correlations")
def wq_alpha_088(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_088(candles, sequential=sequential))


@feature(name="wq_alpha_092", description="WQ101 Alpha #92: Min Decay Correlations")
def wq_alpha_092(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_092(candles, sequential=sequential))


@feature(name="wq_alpha_095", description="WQ101 Alpha #95: Open Range Comparison")
def wq_alpha_095(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_095(candles, sequential=sequential))


@feature(name="wq_alpha_099", description="WQ101 Alpha #99: Correlation Comparison")
def wq_alpha_099(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_099(candles, sequential=sequential))


# ============ Rank-based Alphas ============

@feature(name="wq_alpha_004", description="WQ101 Alpha #4: Low Rank")
def wq_alpha_004(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_004(candles, sequential=sequential))


@feature(name="wq_alpha_007", description="WQ101 Alpha #7: Volume-Close Conditional")
def wq_alpha_007(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_007(candles, sequential=sequential))


@feature(name="wq_alpha_008", description="WQ101 Alpha #8: Open-Returns Product")
def wq_alpha_008(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_008(candles, sequential=sequential))


@feature(name="wq_alpha_010", description="WQ101 Alpha #10: Close Momentum Ranked")
def wq_alpha_010(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_010(candles, sequential=sequential))


@feature(name="wq_alpha_013", description="WQ101 Alpha #13: Close-Volume Covariance")
def wq_alpha_013(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_013(candles, sequential=sequential))


@feature(name="wq_alpha_016", description="WQ101 Alpha #16: High-Volume Covariance")
def wq_alpha_016(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_016(candles, sequential=sequential))


@feature(name="wq_alpha_017", description="WQ101 Alpha #17: Close-Volume Momentum")
def wq_alpha_017(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_017(candles, sequential=sequential))


@feature(name="wq_alpha_019", description="WQ101 Alpha #19: Long-term Returns Momentum")
def wq_alpha_019(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_019(candles, sequential=sequential))


@feature(name="wq_alpha_020", description="WQ101 Alpha #20: Open-Delay Product")
def wq_alpha_020(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_020(candles, sequential=sequential))


@feature(name="wq_alpha_030", description="WQ101 Alpha #30: Sign-based Volume Ratio")
def wq_alpha_030(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_030(candles, sequential=sequential))


@feature(name="wq_alpha_033", description="WQ101 Alpha #33: Open-Close Ratio")
def wq_alpha_033(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_033(candles, sequential=sequential))


@feature(name="wq_alpha_034", description="WQ101 Alpha #34: Returns Volatility Ratio")
def wq_alpha_034(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_034(candles, sequential=sequential))


@feature(name="wq_alpha_035", description="WQ101 Alpha #35: Triple Rank Product")
def wq_alpha_035(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_035(candles, sequential=sequential))


@feature(name="wq_alpha_038", description="WQ101 Alpha #38: Close Rank Ratio")
def wq_alpha_038(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_038(candles, sequential=sequential))


@feature(name="wq_alpha_039", description="WQ101 Alpha #39: Delta Close with Decay Volume")
def wq_alpha_039(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_039(candles, sequential=sequential))


@feature(name="wq_alpha_043", description="WQ101 Alpha #43: Volume-Close Rank Product")
def wq_alpha_043(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_043(candles, sequential=sequential))


@feature(name="wq_alpha_052", description="WQ101 Alpha #52: Returns Momentum with Volume")
def wq_alpha_052(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_052(candles, sequential=sequential))


# ============ VWAP-based Alphas ============

@feature(name="wq_alpha_005", description="WQ101 Alpha #5: Open-VWAP Spread")
def wq_alpha_005(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_005(candles, sequential=sequential))


@feature(name="wq_alpha_011", description="WQ101 Alpha #11: VWAP-Close Range Volume")
def wq_alpha_011(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_011(candles, sequential=sequential))


@feature(name="wq_alpha_025", description="WQ101 Alpha #25: Returns-ADV-VWAP-High")
def wq_alpha_025(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_025(candles, sequential=sequential))


@feature(name="wq_alpha_027", description="WQ101 Alpha #27: Volume-VWAP Correlation")
def wq_alpha_027(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_027(candles, sequential=sequential))


@feature(name="wq_alpha_036", description="WQ101 Alpha #36: Complex Multi-Factor")
def wq_alpha_036(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_036(candles, sequential=sequential))


@feature(name="wq_alpha_041", description="WQ101 Alpha #41: VWAP-HL Spread")
def wq_alpha_041(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_041(candles, sequential=sequential))


@feature(name="wq_alpha_042", description="WQ101 Alpha #42: VWAP-Close Ratio")
def wq_alpha_042(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_042(candles, sequential=sequential))


@feature(name="wq_alpha_047", description="WQ101 Alpha #47: Complex Volume-Price")
def wq_alpha_047(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_047(candles, sequential=sequential))


@feature(name="wq_alpha_050", description="WQ101 Alpha #50: Volume-VWAP Correlation Max")
def wq_alpha_050(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_050(candles, sequential=sequential))


@feature(name="wq_alpha_057", description="WQ101 Alpha #57: Close-VWAP Decay")
def wq_alpha_057(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_057(candles, sequential=sequential))


@feature(name="wq_alpha_061", description="WQ101 Alpha #61: VWAP Rank Comparison")
def wq_alpha_061(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_061(candles, sequential=sequential))


@feature(name="wq_alpha_062", description="WQ101 Alpha #62: VWAP-ADV Correlation")
def wq_alpha_062(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_062(candles, sequential=sequential))


@feature(name="wq_alpha_064", description="WQ101 Alpha #64: Complex Correlation")
def wq_alpha_064(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_064(candles, sequential=sequential))


@feature(name="wq_alpha_065", description="WQ101 Alpha #65: VWAP-ADV vs Open Range")
def wq_alpha_065(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_065(candles, sequential=sequential))


@feature(name="wq_alpha_066", description="WQ101 Alpha #66: VWAP Decay Linear")
def wq_alpha_066(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_066(candles, sequential=sequential))


@feature(name="wq_alpha_071", description="WQ101 Alpha #71: Max Decay Correlations")
def wq_alpha_071(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_071(candles, sequential=sequential))


@feature(name="wq_alpha_072", description="WQ101 Alpha #72: Ratio of Decay Correlations")
def wq_alpha_072(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_072(candles, sequential=sequential))


@feature(name="wq_alpha_073", description="WQ101 Alpha #73: Max VWAP Delta Decay")
def wq_alpha_073(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_073(candles, sequential=sequential))


@feature(name="wq_alpha_074", description="WQ101 Alpha #74: Correlation Comparison")
def wq_alpha_074(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_074(candles, sequential=sequential))


@feature(name="wq_alpha_075", description="WQ101 Alpha #75: VWAP-Volume Correlation")
def wq_alpha_075(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_075(candles, sequential=sequential))


@feature(name="wq_alpha_077", description="WQ101 Alpha #77: Min Decay Correlations")
def wq_alpha_077(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_077(candles, sequential=sequential))


@feature(name="wq_alpha_078", description="WQ101 Alpha #78: Correlation Power")
def wq_alpha_078(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_078(candles, sequential=sequential))


@feature(name="wq_alpha_081", description="WQ101 Alpha #81: VWAP-ADV Correlation Power")
def wq_alpha_081(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_081(candles, sequential=sequential))


@feature(name="wq_alpha_083", description="WQ101 Alpha #83: Complex Ratio")
def wq_alpha_083(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_083(candles, sequential=sequential))


@feature(name="wq_alpha_084", description="WQ101 Alpha #84: VWAP Rank Power")
def wq_alpha_084(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_084(candles, sequential=sequential))


@feature(name="wq_alpha_086", description="WQ101 Alpha #86: ADV Correlation Comparison")
def wq_alpha_086(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_086(candles, sequential=sequential))


@feature(name="wq_alpha_094", description="WQ101 Alpha #94: VWAP Rank Power")
def wq_alpha_094(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_094(candles, sequential=sequential))


@feature(name="wq_alpha_096", description="WQ101 Alpha #96: Max Decay Correlation")
def wq_alpha_096(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_096(candles, sequential=sequential))


@feature(name="wq_alpha_098", description="WQ101 Alpha #98: Decay Correlation Difference")
def wq_alpha_098(candles: np.ndarray, sequential: bool = True) -> np.ndarray:
    return _ensure_array(alpha_098(candles, sequential=sequential))


# ============ All Alpha Names for Reference ============

WQ_ALPHA_FEATURES = [
    # Price/Volume
    "wq_alpha_001", "wq_alpha_009", "wq_alpha_012", "wq_alpha_021",
    "wq_alpha_023", "wq_alpha_024", "wq_alpha_046", "wq_alpha_049",
    "wq_alpha_051", "wq_alpha_053", "wq_alpha_054", "wq_alpha_101",
    # Correlation-based
    "wq_alpha_002", "wq_alpha_003", "wq_alpha_006", "wq_alpha_014",
    "wq_alpha_015", "wq_alpha_018", "wq_alpha_022", "wq_alpha_026",
    "wq_alpha_037", "wq_alpha_040", "wq_alpha_044", "wq_alpha_045",
    "wq_alpha_055", "wq_alpha_068", "wq_alpha_085", "wq_alpha_088",
    "wq_alpha_092", "wq_alpha_095", "wq_alpha_099",
    # Rank-based
    "wq_alpha_004", "wq_alpha_007", "wq_alpha_008", "wq_alpha_010",
    "wq_alpha_013", "wq_alpha_016", "wq_alpha_017", "wq_alpha_019",
    "wq_alpha_020", "wq_alpha_030", "wq_alpha_033", "wq_alpha_034",
    "wq_alpha_035", "wq_alpha_038", "wq_alpha_039", "wq_alpha_043",
    "wq_alpha_052",
    # VWAP-based
    "wq_alpha_005", "wq_alpha_011", "wq_alpha_025", "wq_alpha_027",
    "wq_alpha_036", "wq_alpha_041", "wq_alpha_042", "wq_alpha_047",
    "wq_alpha_050", "wq_alpha_057", "wq_alpha_061", "wq_alpha_062",
    "wq_alpha_064", "wq_alpha_065", "wq_alpha_066", "wq_alpha_071",
    "wq_alpha_072", "wq_alpha_073", "wq_alpha_074", "wq_alpha_075",
    "wq_alpha_077", "wq_alpha_078", "wq_alpha_081", "wq_alpha_083",
    "wq_alpha_084", "wq_alpha_086", "wq_alpha_094", "wq_alpha_096",
    "wq_alpha_098",
]
