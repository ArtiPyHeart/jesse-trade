import json
from pathlib import Path

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)

META_1M = feature_info["meta"]["1m"]
META_3M = feature_info["meta"]["3m"]
META_15M = feature_info["meta"]["15m"]

SIDE_1M = feature_info["side"]["1m"]
SIDE_3M = feature_info["side"]["3m"]
SIDE_15M = feature_info["side"]["15m"]
