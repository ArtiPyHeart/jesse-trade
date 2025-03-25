import json
from pathlib import Path

import lightgbm as lgb

meta_model_path = Path(__file__).parent / "model_meta.txt"
meta_model_prod_path = Path(__file__).parent / "model_meta_prod.txt"


def get_meta_model(is_livetrading: bool):
    if is_livetrading:
        meta_model_prod = lgb.Booster(model_file=meta_model_prod_path)
        return meta_model_prod
    else:
        meta_model = lgb.Booster(model_file=meta_model_path)
        return meta_model


side_model_path = Path(__file__).parent / "model_side.txt"
side_model_prod_path = Path(__file__).parent / "model_side_prod.txt"


def get_side_model(is_livetrading: bool):
    if is_livetrading:
        side_model_prod = lgb.Booster(model_file=side_model_prod_path)
        return side_model_prod
    else:
        side_model = lgb.Booster(model_file=side_model_path)
        return side_model

with open(Path(__file__).parent / "feature_info.json", "r") as f:
    feature_info = json.load(f)

DOLLAR_BAR_SHORT_TERM = "30m"
DOLLAR_BAR_MID_TERM = "2h"
DOLLAR_BAR_LONG_TERM = "6h"

SIDE_DOLLAR_BAR_SHORT_FEATURES = feature_info["side"][DOLLAR_BAR_SHORT_TERM]
SIDE_DOLLAR_BAR_MID_FEATURES = feature_info["side"][DOLLAR_BAR_MID_TERM]
SIDE_DOLLAR_BAR_LONG_FEATURES = feature_info["side"][DOLLAR_BAR_LONG_TERM]

SIDE_DOLLAR_BAR_ALL = (
    SIDE_DOLLAR_BAR_SHORT_FEATURES
    + SIDE_DOLLAR_BAR_MID_FEATURES
    + SIDE_DOLLAR_BAR_LONG_FEATURES
)
SIDE_ALL = SIDE_DOLLAR_BAR_ALL

META_DOLLAR_BAR_SHORT_FEATURES = feature_info["meta"][DOLLAR_BAR_SHORT_TERM]
META_DOLLAR_BAR_MID_FEATURES = feature_info["meta"][DOLLAR_BAR_MID_TERM]
META_DOLLAR_BAR_LONG_FEATURES = feature_info["meta"][DOLLAR_BAR_LONG_TERM]
META_MODEL_SIDE_RES = feature_info["meta"]["model_res"]


META_DOLLAR_BAR_ALL = (
    META_DOLLAR_BAR_SHORT_FEATURES
    + META_DOLLAR_BAR_MID_FEATURES
    + META_DOLLAR_BAR_LONG_FEATURES
)
META_ALL = META_DOLLAR_BAR_ALL + META_MODEL_SIDE_RES

DOLLAR_BAR_THRESHOLD_SHORT = 248550043.9622903
DOLLAR_BAR_THRESHOLD_MID = 1052682535.8991117
DOLLAR_BAR_THRESHOLD_LONG = 3216529968.747286
