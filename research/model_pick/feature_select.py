import gc
import hashlib

import numpy as np
import pandas as pd
from pathlib import Path

from src.features.feature_selection.rfcq_selector import RFCQSelector
from src.models.deep_ssm import DeepSSMConfig, DeepSSM
from src.models.lgssm import LGSSMConfig, LGSSM
from .features import ALL_FEATS

FRAC_FEATS = [i for i in ALL_FEATS if i.startswith("frac_") and i.endswith("_diff")]
deep_ssm_config = DeepSSMConfig(
    obs_dim=len(FRAC_FEATS),
)
lg_ssm_config = LGSSMConfig(
    obs_dim=len(FRAC_FEATS),
)


class FeatureSelector:
    def __init__(self, model_save_dir: Path = None, load_existing: bool = False):
        self.model_save_dir = model_save_dir
        # 🔧 添加缓存，避免重复计算大型 DataFrame
        self._cached_all_features = None
        self._cached_train_x_hash = None  # 使用内容哈希替代 id()，更可靠

        if load_existing and model_save_dir:
            # 尝试加载已有模型
            deep_ssm_path = model_save_dir / "deep_ssm"
            lg_ssm_path = model_save_dir / "lg_ssm"

            if deep_ssm_path.with_suffix(".safetensors").exists():
                self.deep_ssm_model = DeepSSM.load(deep_ssm_path.resolve().as_posix())
            else:
                self.deep_ssm_model = DeepSSM(config=deep_ssm_config)

            if lg_ssm_path.with_suffix(".safetensors").exists():
                self.lg_ssm_model = LGSSM.load(lg_ssm_path.resolve().as_posix())
            else:
                self.lg_ssm_model = LGSSM(config=lg_ssm_config)
        else:
            self.deep_ssm_model = DeepSSM(config=deep_ssm_config)
            self.lg_ssm_model = LGSSM(config=lg_ssm_config)

    @property
    def selector(self):
        return RFCQSelector(verbose=True)

    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """基于内容的快速哈希，用于可靠的缓存检测"""
        # 使用 shape + 首尾行 + 首个索引值进行快速哈希
        shape_str = f"{df.shape}"
        first_row = df.iloc[0].values.tobytes() if len(df) > 0 else b""
        last_row = df.iloc[-1].values.tobytes() if len(df) > 0 else b""
        index_val = str(df.index[0]) if len(df) > 0 else "0"
        content = shape_str.encode() + first_row + last_row + index_val.encode()
        return hashlib.md5(content).hexdigest()

    def fit(self, train_x):
        if not self.deep_ssm_model.is_fitted:
            self.deep_ssm_model.fit(train_x[FRAC_FEATS])
            # 保存 deep ssm 模型
            if self.model_save_dir:
                self.model_save_dir.mkdir(parents=True, exist_ok=True)
                deep_ssm_path = self.model_save_dir / "deep_ssm"
                self.deep_ssm_model.save(deep_ssm_path.resolve().as_posix())

        if not self.lg_ssm_model.is_fitted:
            self.lg_ssm_model.fit(train_x[FRAC_FEATS])
            # 保存 lg ssm 模型
            if self.model_save_dir:
                self.model_save_dir.mkdir(parents=True, exist_ok=True)
                lg_ssm_path = self.model_save_dir / "lg_ssm"
                self.lg_ssm_model.save(lg_ssm_path.resolve().as_posix())

    def get_deep_ssm_features(self, train_x):
        feat_deep_ssm = self.deep_ssm_model.transform(train_x[FRAC_FEATS])
        df_deep_ssm = pd.DataFrame(
            feat_deep_ssm,
            columns=[f"deep_ssm_{i}" for i in range(feat_deep_ssm.shape[1])],
            index=train_x.index,
        )
        return df_deep_ssm

    def get_lg_ssm_features(self, train_x):
        feat_lg_ssm = self.lg_ssm_model.transform(train_x[FRAC_FEATS])
        df_lg_ssm = pd.DataFrame(
            feat_lg_ssm,
            columns=[f"lg_ssm_{i}" for i in range(feat_lg_ssm.shape[1])],
            index=train_x.index,
        )
        return df_lg_ssm

    def get_all_features(self, train_x):
        # 🔧 使用内容哈希进行可靠的缓存检测（替代不可靠的 id()）
        current_hash = self._compute_data_hash(train_x)
        if (
            self._cached_all_features is not None
            and self._cached_train_x_hash == current_hash
        ):
            return self._cached_all_features

        self.fit(train_x)
        df_deep_ssm = self.get_deep_ssm_features(train_x)
        lg_ssm_features = self.get_lg_ssm_features(train_x)

        # 🔧 使用预分配 numpy 数组替代 pd.concat，减少内存分配
        n_rows = len(train_x)
        n_cols_total = (
            df_deep_ssm.shape[1] + lg_ssm_features.shape[1] + train_x.shape[1]
        )

        # 预分配结果数组（使用 float32 节省内存）
        result_data = np.empty((n_rows, n_cols_total), dtype=np.float32)

        # 直接赋值，避免中间副本
        col_offset = 0
        result_data[:, col_offset : col_offset + df_deep_ssm.shape[1]] = (
            df_deep_ssm.values
        )
        col_offset += df_deep_ssm.shape[1]
        result_data[:, col_offset : col_offset + lg_ssm_features.shape[1]] = (
            lg_ssm_features.values
        )
        col_offset += lg_ssm_features.shape[1]
        result_data[:, col_offset:] = train_x.values

        # 构建列名列表
        columns = (
            list(df_deep_ssm.columns)
            + list(lg_ssm_features.columns)
            + list(train_x.columns)
        )

        df = pd.DataFrame(result_data, index=train_x.index, columns=columns)

        # 显式删除中间对象，释放内存
        del df_deep_ssm, lg_ssm_features, result_data
        gc.collect()

        # 缓存结果
        self._cached_all_features = df
        self._cached_train_x_hash = current_hash
        return df

    def get_all_features_no_fit(self, train_x):
        """获取所有特征但不重新训练模型，适用于已加载模型的情况"""
        df_deep_ssm = self.get_deep_ssm_features(train_x)
        lg_ssm_features = self.get_lg_ssm_features(train_x)

        # 🔧 使用预分配数组替代 pd.concat
        n_rows = len(train_x)
        n_cols_total = (
            df_deep_ssm.shape[1] + lg_ssm_features.shape[1] + train_x.shape[1]
        )
        result_data = np.empty((n_rows, n_cols_total), dtype=np.float32)

        col_offset = 0
        result_data[:, col_offset : col_offset + df_deep_ssm.shape[1]] = (
            df_deep_ssm.values
        )
        col_offset += df_deep_ssm.shape[1]
        result_data[:, col_offset : col_offset + lg_ssm_features.shape[1]] = (
            lg_ssm_features.values
        )
        col_offset += lg_ssm_features.shape[1]
        result_data[:, col_offset:] = train_x.values

        columns = (
            list(df_deep_ssm.columns)
            + list(lg_ssm_features.columns)
            + list(train_x.columns)
        )
        df = pd.DataFrame(result_data, index=train_x.index, columns=columns)

        del df_deep_ssm, lg_ssm_features, result_data
        return df

    def select_features(self, train_x, train_y) -> list[str]:
        _selector = self.selector
        df_feat = self.get_all_features(train_x)
        _selector.fit(df_feat, train_y)
        res = pd.Series(_selector.relevance_, index=_selector.variables_).sort_values(
            ascending=False
        )
        feature_names = res[res > 0].index.tolist()
        return feature_names

    def clear_cache(self):
        """🔧 清理缓存的特征数据和模型状态，释放内存"""
        self._cached_all_features = None
        self._cached_train_x_hash = None

        # 清理 SSM 模型的梯度缓存（如果存在）
        if hasattr(self, "deep_ssm_model") and self.deep_ssm_model is not None:
            if hasattr(self.deep_ssm_model, "model") and hasattr(
                self.deep_ssm_model.model, "zero_grad"
            ):
                self.deep_ssm_model.model.zero_grad(set_to_none=True)

        if hasattr(self, "lg_ssm_model") and self.lg_ssm_model is not None:
            if hasattr(self.lg_ssm_model, "model") and hasattr(
                self.lg_ssm_model.model, "zero_grad"
            ):
                self.lg_ssm_model.model.zero_grad(set_to_none=True)

        gc.collect()
