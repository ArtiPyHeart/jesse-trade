from pathlib import Path

from custom_indicators.utils.import_tools import ensure_package

ensure_package("lightgbm")
import lightgbm as lgb  # noqa: E402

path_meta_model = Path(__file__).parent / "model" / "model_meta.txt"
path_meta_model_prod = Path(__file__).parent / "model" / "model_meta_prod.txt"


def get_meta_model(is_livetrading: bool):
    if is_livetrading:
        meta_model_prod = lgb.Booster(model_file=path_meta_model_prod)
        return meta_model_prod
    else:
        meta_model = lgb.Booster(model_file=path_meta_model)
        return meta_model


path_side_model_long = Path(__file__).parent / "model" / "model_side_long.txt"
path_side_model_long_prod = Path(__file__).parent / "model" / "model_side_long_prod.txt"


def get_side_model(is_livetrading: bool):
    if is_livetrading:
        side_model_prod = lgb.Booster(model_file=path_side_model_long_prod)
        return side_model_prod
    else:
        side_model = lgb.Booster(model_file=path_side_model_long)
        return side_model


with open(Path(__file__).parent / "feature_info.json", "r") as f:
    import json

    feature_info = json.load(f)

SIDE = feature_info["side"]["side"]

META_FEATURES = feature_info["meta"]["meta"]
META_MODEL = feature_info["meta"]["model"]
META_ALL = META_FEATURES + META_MODEL
