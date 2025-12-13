"""
SSM Adapters 单元测试

测试 DeepSSMAdapter 和 LGSSMAdapter 的接口一致性和正确性。
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.ssm import SSMProtocol, DeepSSMAdapter, LGSSMAdapter
from src.models.deep_ssm import DeepSSMConfig
from src.models.lgssm import LGSSMConfig


class TestSSMProtocol:
    """测试 SSMProtocol 实现"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        # 创建带有一定结构的数据
        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    def test_deep_ssm_adapter_implements_protocol(self):
        """DeepSSMAdapter 应该符合 SSMProtocol"""
        config = DeepSSMConfig(obs_dim=5, state_dim=3, max_epochs=5)
        adapter = DeepSSMAdapter(config=config)
        assert isinstance(adapter, SSMProtocol)

    def test_lg_ssm_adapter_implements_protocol(self):
        """LGSSMAdapter 应该符合 SSMProtocol"""
        config = LGSSMConfig(obs_dim=5, state_dim=3, max_epochs=5)
        adapter = LGSSMAdapter(config=config)
        assert isinstance(adapter, SSMProtocol)


class TestDeepSSMAdapter:
    """DeepSSMAdapter 单元测试"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    @pytest.fixture
    def fitted_adapter(self, sample_data) -> DeepSSMAdapter:
        """创建并训练适配器"""
        config = DeepSSMConfig(
            obs_dim=sample_data.shape[1],
            state_dim=3,
            max_epochs=10,
            patience=3,
        )
        adapter = DeepSSMAdapter(config=config)
        adapter.fit(sample_data)
        return adapter

    def test_fit_transform(self, sample_data):
        """测试 fit 和 transform"""
        config = DeepSSMConfig(
            obs_dim=sample_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        adapter = DeepSSMAdapter(config=config)

        assert not adapter.is_fitted

        adapter.fit(sample_data)

        assert adapter.is_fitted
        assert adapter.state_dim == 3
        assert adapter.obs_dim == sample_data.shape[1]
        assert adapter.prefix == "deep_ssm"

        # transform
        result = adapter.transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert result.shape[1] == 3
        assert all(col.startswith("deep_ssm_") for col in result.columns)

    def test_inference(self, fitted_adapter, sample_data):
        """测试单步推理"""
        obs = sample_data.iloc[0].values

        state = fitted_adapter.inference(obs)

        assert isinstance(state, np.ndarray)
        assert state.shape == (fitted_adapter.state_dim,)

    def test_reset_state(self, fitted_adapter, sample_data):
        """测试状态重置"""
        obs = sample_data.iloc[0].values

        # 推理几步
        for i in range(5):
            fitted_adapter.inference(sample_data.iloc[i].values)

        # 重置
        fitted_adapter.reset_state()

        # 重新推理应该得到相同结果
        state1 = fitted_adapter.inference(obs)

        fitted_adapter.reset_state()
        state2 = fitted_adapter.inference(obs)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_inference_vs_transform_consistency(self, fitted_adapter, sample_data):
        """测试单步推理与批量转换的数值一致性"""
        # 重置状态
        fitted_adapter.reset_state()

        # 逐行推理
        inference_results = []
        for i in range(len(sample_data)):
            state = fitted_adapter.inference(sample_data.iloc[i].values)
            inference_results.append(state.copy())

        inference_array = np.array(inference_results)

        # 批量转换
        transform_result = fitted_adapter.transform(sample_data)

        # 比较结果
        np.testing.assert_array_almost_equal(
            inference_array,
            transform_result.values,
            decimal=5,
            err_msg="DeepSSM inference and transform results should be consistent",
        )

    def test_save_load(self, fitted_adapter, sample_data):
        """测试保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "deep_ssm"

            # 保存
            fitted_adapter.save(str(save_path))

            # 验证文件存在
            assert save_path.with_suffix(".safetensors").exists()
            assert save_path.with_suffix(".json").exists()

            # 加载
            loaded = DeepSSMAdapter.load(str(save_path))

            assert loaded.is_fitted
            assert loaded.state_dim == fitted_adapter.state_dim
            assert loaded.obs_dim == fitted_adapter.obs_dim

            # 验证 transform 结果一致
            result_original = fitted_adapter.transform(sample_data)
            result_loaded = loaded.transform(sample_data)

            pd.testing.assert_frame_equal(result_original, result_loaded)


class TestLGSSMAdapter:
    """LGSSMAdapter 单元测试"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """创建测试数据"""
        np.random.seed(42)
        n_samples = 200
        n_features = 5

        t = np.linspace(0, 4 * np.pi, n_samples)
        data = np.column_stack(
            [np.sin(t + i * np.pi / n_features) for i in range(n_features)]
        )
        data += np.random.randn(n_samples, n_features) * 0.1

        return pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    @pytest.fixture
    def fitted_adapter(self, sample_data) -> LGSSMAdapter:
        """创建并训练适配器"""
        config = LGSSMConfig(
            obs_dim=sample_data.shape[1],
            state_dim=3,
            max_epochs=10,
            patience=3,
        )
        adapter = LGSSMAdapter(config=config)
        adapter.fit(sample_data)
        return adapter

    def test_fit_transform(self, sample_data):
        """测试 fit 和 transform"""
        config = LGSSMConfig(
            obs_dim=sample_data.shape[1],
            state_dim=3,
            max_epochs=5,
        )
        adapter = LGSSMAdapter(config=config)

        assert not adapter.is_fitted

        adapter.fit(sample_data)

        assert adapter.is_fitted
        assert adapter.state_dim == 3
        assert adapter.obs_dim == sample_data.shape[1]
        assert adapter.prefix == "lg_ssm"

        # transform
        result = adapter.transform(sample_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert result.shape[1] == 3
        assert all(col.startswith("lg_ssm_") for col in result.columns)

    def test_inference(self, fitted_adapter, sample_data):
        """测试单步推理"""
        obs = sample_data.iloc[0].values

        state = fitted_adapter.inference(obs)

        assert isinstance(state, np.ndarray)
        assert state.shape == (fitted_adapter.state_dim,)

    def test_inference_vs_transform_consistency(self, fitted_adapter, sample_data):
        """测试单步推理与批量转换的一致性"""
        # 重置状态
        fitted_adapter.reset_state()

        # 逐行推理
        inference_results = []
        for i in range(len(sample_data)):
            state = fitted_adapter.inference(sample_data.iloc[i].values)
            inference_results.append(state.copy())

        inference_array = np.array(inference_results)

        # 批量转换
        transform_result = fitted_adapter.transform(sample_data)

        # 比较结果
        np.testing.assert_array_almost_equal(
            inference_array,
            transform_result.values,
            decimal=5,
            err_msg="Inference and transform results should be consistent",
        )

    def test_reset_state(self, fitted_adapter, sample_data):
        """测试状态重置"""
        obs = sample_data.iloc[0].values

        # 推理几步
        for i in range(5):
            fitted_adapter.inference(sample_data.iloc[i].values)

        # 重置
        fitted_adapter.reset_state()

        # 重新推理应该得到相同结果
        state1 = fitted_adapter.inference(obs)

        fitted_adapter.reset_state()
        state2 = fitted_adapter.inference(obs)

        np.testing.assert_array_almost_equal(state1, state2)

    def test_save_load(self, fitted_adapter, sample_data):
        """测试保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "lg_ssm"

            # 保存
            fitted_adapter.save(str(save_path))

            # 验证文件存在
            assert save_path.with_suffix(".safetensors").exists()
            assert save_path.with_suffix(".json").exists()

            # 加载
            loaded = LGSSMAdapter.load(str(save_path))

            assert loaded.is_fitted
            assert loaded.state_dim == fitted_adapter.state_dim
            assert loaded.obs_dim == fitted_adapter.obs_dim

            # 验证 transform 结果一致
            result_original = fitted_adapter.transform(sample_data)
            result_loaded = loaded.transform(sample_data)

            pd.testing.assert_frame_equal(result_original, result_loaded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
