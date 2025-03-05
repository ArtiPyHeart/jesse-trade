import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)


SIDE_3M = feature_info["side"]["10m"]
SIDE_15M = feature_info["side"]["45m"]
SIDE_1H = feature_info["side"]["4h"]

SIDE_ALL = SIDE_3M + SIDE_15M + SIDE_1H

META_3M = feature_info["meta"]["10m"]
META_15M = feature_info["meta"]["45m"]
META_1H = feature_info["meta"]["4h"]
META_MODEL_RES = feature_info["meta"]["model_res"]

META_ALL = META_3M + META_15M + META_1H + META_MODEL_RES

DOLLAR_BAR_THRESHOLD_10M = 71968406.59988706
DOLLAR_BAR_THRESHOLD_45M = 379150625.6237953
DOLLAR_BAR_THRESHOLD_4H = 2078307128.56747
