import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)


SIDE_3M = feature_info["side"]["3m"]
SIDE_15M = feature_info["side"]["15m"]
SIDE_1H = feature_info["side"]["1h"]

SIDE_ALL = SIDE_3M + SIDE_15M + SIDE_1H

META_3M = feature_info["meta"]["3m"]
META_15M = feature_info["meta"]["15m"]
META_1H = feature_info["meta"]["1h"]
META_MODEL_RES = feature_info["meta"]["model_res"]

META_ALL = META_3M + META_15M + META_1H + META_MODEL_RES

DOLLAR_BAR_THRESHOLD_5M = 30718222.902390815
DOLLAR_BAR_THRESHOLD_15M = 115851580.74615964
DOLLAR_BAR_THRESHOLD_45M = 379150625.6237953
DOLLAR_BAR_THRESHOLD_1H = 505534167.1650603
