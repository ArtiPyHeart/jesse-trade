from pathlib import Path

import lightgbm as lgb

meta_model_path = Path(__file__).parent / "model" / "model_meta.txt"
meta_model_prod_path = Path(__file__).parent / "model" / "model_meta_prod.txt"


def get_meta_model(is_livetrading: bool):
    if is_livetrading:
        meta_model_prod = lgb.Booster(model_file=meta_model_prod_path)
        return meta_model_prod
    else:
        meta_model = lgb.Booster(model_file=meta_model_path)
        return meta_model


path_side_model_long = Path(__file__).parent / "model" / "model_side_long.txt"
path_side_model_short = Path(__file__).parent / "model" / "model_side_short.txt"


def get_side_model(is_livetrading: bool, side: str):
    if is_livetrading:
        raise NotImplementedError
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
META_MODEL_RES = feature_info["meta"]["model_res"]
META_ALL = META_FEATURES + META_MODEL_RES
