import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)


SIDE_10M = feature_info["side"]["10m"]
SIDE_25M = feature_info["side"]["25m"]
SIDE_2H = feature_info["side"]["2h"]

SIDE_ALL = SIDE_10M + SIDE_25M + SIDE_2H

META_10M = feature_info["meta"]["10m"]
META_25M = feature_info["meta"]["25m"]
META_2H = feature_info["meta"]["2h"]
META_MODEL_RES = feature_info["meta"]["model_res"]

META_ALL = META_10M + META_25M + META_2H + META_MODEL_RES

DOLLAR_BAR_THRESHOLD_10M = 71968406.59988706
DOLLAR_BAR_THRESHOLD_25M = 203617929.03870487
DOLLAR_BAR_THRESHOLD_2H = 1039153564.783735
