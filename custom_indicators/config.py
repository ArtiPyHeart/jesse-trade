import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)


SIDE_15M = feature_info["side"]["15m"]
SIDE_45M = feature_info["side"]["45m"]
SIDE_4H = feature_info["side"]["4h"]

SIDE_ALL = SIDE_15M + SIDE_45M + SIDE_4H

META_15M = feature_info["meta"]["15m"]
META_45M = feature_info["meta"]["45m"]
META_4H = feature_info["meta"]["4h"]
META_MODEL_RES = feature_info["meta"]["model_res"]

META_ALL = META_15M + META_45M + META_4H + META_MODEL_RES

DOLLAR_BAR_THRESHOLD_15M = 115851580.74615964
DOLLAR_BAR_THRESHOLD_45M = 379150625.6237953
DOLLAR_BAR_THRESHOLD_4H = 2078307128.56747
