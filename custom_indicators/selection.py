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
