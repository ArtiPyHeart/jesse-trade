from pathlib import Path

from custom_indicators.utils.import_tools import ensure_package

ensure_package("lightgbm")
import lightgbm as lgb

MIN_WINDOW = 20
MAX_WINDOW = 60 * 24
WINDOW = 134
VOL_T_WINDOW = 78
VOL_REF_WINDOW = 1341
ENTROPY_THRESHOLD = 26.96225407898441

path_meta_model = Path(__file__).parent / "model" / "model_meta.txt"


def get_meta_model(is_livetrading: bool):
    if is_livetrading:
        meta_model_prod = lgb.Booster(model_file=path_meta_model)
        return meta_model_prod
    else:
        meta_model = lgb.Booster(model_file=path_meta_model)
        return meta_model


path_side_model_long = Path(__file__).parent / "model" / "model_side_long.txt"
path_side_model_short = Path(__file__).parent / "model" / "model_side_short.txt"


def get_side_model(is_livetrading: bool, side: str):
    if is_livetrading:
        if side == "long":
            side_model_prod = lgb.Booster(model_file=path_side_model_long)
        elif side == "short":
            side_model_prod = lgb.Booster(model_file=path_side_model_short)
        else:
            raise ValueError(f"Invalid side: {side}")
        return side_model_prod
    else:
        if side == "long":
            side_model = lgb.Booster(model_file=path_side_model_long)
        elif side == "short":
            side_model = lgb.Booster(model_file=path_side_model_short)
        else:
            raise ValueError(f"Invalid side: {side}")
        return side_model


with open(Path(__file__).parent / "feature_info.json", "r") as f:
    import json

    feature_info = json.load(f)

SIDE_LONG = feature_info["side"]["long"]
SIDE_SHORT = feature_info["side"]["short"]

META_FEATURES = feature_info["meta"]["meta"]
META_MODEL_LONG = feature_info["meta"]["model_long"]
META_MODEL_SHORT = feature_info["meta"]["model_short"]
META_ALL = META_FEATURES + META_MODEL_LONG + META_MODEL_SHORT
