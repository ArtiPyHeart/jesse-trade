import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)

SHORT_TERM = "15m"
MID_TERM = "1h"
LONG_TERM = "4h"

SIDE_SHORT = feature_info["side"][SHORT_TERM]
SIDE_MID = feature_info["side"][MID_TERM]
SIDE_LONG = feature_info["side"][LONG_TERM]

SIDE_ALL = SIDE_SHORT + SIDE_MID + SIDE_LONG

META_SHORT = feature_info["meta"][SHORT_TERM]
META_MID = feature_info["meta"][MID_TERM]
META_LONG = feature_info["meta"][LONG_TERM]
META_MODEL_RES = feature_info["meta"]["model_res"]

META_ALL = META_SHORT + META_MID + META_LONG + META_MODEL_RES

DOLLAR_BAR_THRESHOLD_SHORT = 39951050.90908544
DOLLAR_BAR_THRESHOLD_MID = 111681345.0640343
DOLLAR_BAR_THRESHOLD_LONG = 279657350.36359805

RANGE_BAR_THRESHOLD_SHORT = 0.0045
RANGE_BAR_THRESHOLD_MID = 0.009
RANGE_BAR_THRESHOLD_LONG = 0.018
