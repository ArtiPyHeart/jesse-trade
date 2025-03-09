import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)

SHORT_TERM = "10m"
MID_TERM = "25m"
LONG_TERM = "2h"

SIDE_SHORT = feature_info["side"][SHORT_TERM]
SIDE_MID = feature_info["side"][MID_TERM]
SIDE_LONG = feature_info["side"][LONG_TERM]

SIDE_ALL = SIDE_SHORT + SIDE_MID + SIDE_LONG

META_SHORT = feature_info["meta"][SHORT_TERM]
META_MID = feature_info["meta"][MID_TERM]
META_LONG = feature_info["meta"][LONG_TERM]
META_MODEL_RES = feature_info["meta"]["model_res"]

META_ALL = META_SHORT + META_MID + META_LONG + META_MODEL_RES

DOLLAR_BAR_THRESHOLD_SHORT = 71968406.59988706
DOLLAR_BAR_THRESHOLD_MID = 203617929.03870487
DOLLAR_BAR_THRESHOLD_LONG = 1039153564.783735
