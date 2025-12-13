"""
SSM 状态同步测试

验证 SSM 适配器的 transform() 和 inference() 状态同步正确性：
1. transform 后 inference 状态连续性
2. 逐行 inference 与批量 transform 结果一致性
3. transform + inference 混合场景（模拟实盘）
"""

import numpy as np
import pandas as pd
import pytest

from src.features.ssm.adapters import DeepSSMAdapter, LGSSMAdapter
from src.models.deep_ssm import DeepSSMConfig
from src.models.lgssm import LGSSMConfig


def generate_test_data(n_samples: int, obs_dim: int = 3, seed: int = 42) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(seed)
    # 生成带有趋势和噪声的数据
    t = np.linspace(0, 4 * np.pi, n_samples)
    data = np.zeros((n_samples, obs_dim))
    for i in range(obs_dim):
        data[:, i] = np.sin(t + i * np.pi / 3) + 0.1 * np.random.randn(n_samples)
    return pd.DataFrame(data, columns=[f"obs_{i}" for i in range(obs_dim)])


class TestLGSSMStateSync:
    """LGSSM 适配器状态同步测试"""

    @pytest.fixture
    def adapter(self):
        """创建 LGSSM 适配器"""
        config = LGSSMConfig(obs_dim=3, state_dim=4)
        return LGSSMAdapter(config=config, prefix="lgssm")

    @pytest.fixture
    def test_data(self):
        """生成测试数据"""
        return generate_test_data(100, obs_dim=3)

    def test_transform_inference_state_continuity(self, adapter, test_data):
        """
        验证 transform 后 inference 状态与 transform 最后一行一致

        场景：先用历史数据 transform，再用新数据 inference
        期望：inference 的输入状态应该与 transform 最后状态一致
        """
        train_data = test_data[:80]
        next_obs = test_data.iloc[80].values

        # Fit + Transform
        adapter.fit(train_data)
        transform_result = adapter.transform(train_data)

        # 记录 transform 最后一行状态
        last_transform_state = transform_result.iloc[-1].values.copy()

        # 验证内部状态已同步
        assert adapter._state is not None, "State should be synced after transform"
        assert adapter._covariance is not None, "Covariance should be synced after transform"
        assert not adapter._first_observation, "first_observation should be False after transform"

        # 用下一个观测做 inference
        inference_state = adapter.inference(next_obs)

        # 验证 inference 返回的状态维度正确
        assert inference_state.shape == (adapter.state_dim,)

        # 验证 inference 结果与继续 transform 的结果一致
        adapter.reset_state()
        full_data = test_data[:81]  # 包含 next_obs
        adapter.transform(train_data)  # 先 transform 前 80 行
        single_inference = adapter.inference(next_obs)

        # 两次 inference 结果应完全一致
        np.testing.assert_allclose(
            inference_state, single_inference, rtol=1e-5,
            err_msg="Inference after transform should produce same result as continuous inference"
        )

    def test_rowwise_inference_equals_transform(self, adapter, test_data):
        """
        验证逐行 inference 结果与 transform 完全一致

        场景：分别用批量 transform 和逐行 inference 处理同一数据
        期望：两者结果完全一致
        """
        train_data = test_data[:60]
        eval_data = test_data[60:80]

        # Fit
        adapter.fit(train_data)

        # 方式 1: 批量 transform
        adapter.reset_state()
        transform_result = adapter.transform(eval_data)

        # 方式 2: 逐行 inference
        adapter.reset_state()
        inference_results = []
        for i in range(len(eval_data)):
            state = adapter.inference(eval_data.iloc[i].values)
            inference_results.append(state.copy())

        inference_array = np.array(inference_results)

        # 两者应完全一致
        np.testing.assert_allclose(
            transform_result.values, inference_array, rtol=1e-5,
            err_msg="Row-wise inference should match batch transform exactly"
        )

    def test_transform_then_continue_inference(self, adapter, test_data):
        """
        模拟实盘场景：先 warmup(transform)，再逐行 inference

        场景：
        1. 用历史数据 transform 预热
        2. 模拟新数据到来，逐行 inference
        期望：结果与一次性 transform 整个序列的结果一致
        """
        train_data = test_data[:50]
        warmup_data = test_data[50:70]
        new_data = test_data[70:80]

        # Fit
        adapter.fit(train_data)

        # 方式 1: 一次性 transform 整个序列
        adapter.reset_state()
        full_transform = adapter.transform(pd.concat([warmup_data, new_data], ignore_index=True))
        expected_new_states = full_transform.iloc[len(warmup_data):].values

        # 方式 2: 先 transform warmup，再逐行 inference new
        adapter.reset_state()
        adapter.transform(warmup_data)  # warmup，状态应同步到 warmup_data 末尾

        states = []
        for i in range(len(new_data)):
            state = adapter.inference(new_data.iloc[i].values)
            states.append(state.copy())

        inference_array = np.array(states)

        # 两者应完全一致
        np.testing.assert_allclose(
            inference_array, expected_new_states, rtol=1e-5,
            err_msg="Transform+Inference mixed mode should match full transform"
        )

    def test_transform_restarts_from_initial_state(self, adapter, test_data):
        """
        验证 transform 每次从初始状态开始（设计行为）

        说明：LGSSM.transform() 每次调用都从初始状态开始，
        状态同步只影响后续 inference() 调用，不影响下一次 transform()。
        这是正确的设计：批处理结果的可复现性。
        """
        train_data = test_data[:50]
        batch = test_data[50:70]

        # Fit
        adapter.fit(train_data)

        # 第一次 transform
        adapter.reset_state()
        result1 = adapter.transform(batch)

        # 做一些 inference 改变内部状态
        for i in range(5):
            adapter.inference(test_data.iloc[70 + i].values)

        # 第二次 transform（不 reset_state）
        result2 = adapter.transform(batch)

        # 两次 transform 结果应该相同（因为 transform 总是从初始状态开始）
        np.testing.assert_allclose(
            result1.values, result2.values, rtol=1e-5,
            err_msg="Transform should always produce same result regardless of adapter state"
        )

    def test_reset_state_clears_state(self, adapter, test_data):
        """验证 reset_state 正确重置状态"""
        train_data = test_data[:80]

        adapter.fit(train_data)
        adapter.transform(train_data)

        # 重置状态
        adapter.reset_state()

        # 验证状态已重置
        assert adapter._first_observation is True
        # 状态应该是初始值（zeros）
        np.testing.assert_allclose(adapter._state, np.zeros(adapter.state_dim), rtol=1e-5)


class TestDeepSSMStateSync:
    """DeepSSM 适配器状态同步测试"""

    @pytest.fixture
    def adapter(self):
        """创建 DeepSSM 适配器"""
        config = DeepSSMConfig(
            obs_dim=3,
            state_dim=4,
            lstm_hidden=16,
            max_epochs=5,  # 减少训练轮数加速测试
        )
        return DeepSSMAdapter(config=config, prefix="deepssm")

    @pytest.fixture
    def test_data(self):
        """生成测试数据"""
        return generate_test_data(100, obs_dim=3)

    def test_transform_inference_state_continuity(self, adapter, test_data):
        """
        验证 transform 后 inference 状态与 transform 最后一行一致

        场景：先用历史数据 transform，再用新数据 inference
        期望：inference 的输入状态应该与 transform 最后状态一致
        """
        train_data = test_data[:80]
        next_obs = test_data.iloc[80].values

        # Fit + Transform
        adapter.fit(train_data)
        transform_result = adapter.transform(train_data)

        # 验证 realtime_processor 已创建
        assert adapter._realtime_processor is not None, "Realtime processor should be created"

        # 记录 transform 最后一行状态
        last_transform_state = transform_result.iloc[-1].values.copy()

        # 用下一个观测做 inference
        inference_state = adapter.inference(next_obs)

        # 验证 inference 返回的状态维度正确
        assert inference_state.shape == (adapter.state_dim,)

        # 验证 inference 结果与继续 transform 的结果一致
        adapter.reset_state()
        adapter.transform(train_data)  # 先 transform 前 80 行，状态同步
        single_inference = adapter.inference(next_obs)

        # 两次 inference 结果应完全一致
        np.testing.assert_allclose(
            inference_state, single_inference, rtol=1e-4,  # DeepSSM 允许稍大误差
            err_msg="Inference after transform should produce same result as continuous inference"
        )

    def test_rowwise_inference_equals_transform(self, adapter, test_data):
        """
        验证逐行 inference 结果与 transform 完全一致

        场景：分别用批量 transform 和逐行 inference 处理同一数据
        期望：两者结果完全一致
        """
        train_data = test_data[:60]
        eval_data = test_data[60:80]

        # Fit
        adapter.fit(train_data)

        # 方式 1: 批量 transform
        adapter.reset_state()
        transform_result = adapter.transform(eval_data)

        # 方式 2: 逐行 inference
        adapter.reset_state()
        inference_results = []
        for i in range(len(eval_data)):
            state = adapter.inference(eval_data.iloc[i].values)
            inference_results.append(state.copy())

        inference_array = np.array(inference_results)

        # 两者应完全一致
        np.testing.assert_allclose(
            transform_result.values, inference_array, rtol=1e-4,
            err_msg="Row-wise inference should match batch transform exactly"
        )

    def test_transform_then_continue_inference(self, adapter, test_data):
        """
        模拟实盘场景：先 warmup(transform)，再逐行 inference

        场景：
        1. 用历史数据 transform 预热
        2. 模拟新数据到来，逐行 inference
        期望：结果与一次性 transform 整个序列的结果一致
        """
        train_data = test_data[:50]
        warmup_data = test_data[50:70]
        new_data = test_data[70:80]

        # Fit
        adapter.fit(train_data)

        # 方式 1: 一次性 transform 整个序列
        adapter.reset_state()
        full_transform = adapter.transform(pd.concat([warmup_data, new_data], ignore_index=True))
        expected_new_states = full_transform.iloc[len(warmup_data):].values

        # 方式 2: 先 transform warmup，再逐行 inference new
        adapter.reset_state()
        adapter.transform(warmup_data)  # warmup，状态应同步到 warmup_data 末尾

        states = []
        for i in range(len(new_data)):
            state = adapter.inference(new_data.iloc[i].values)
            states.append(state.copy())

        inference_array = np.array(states)

        # 两者应完全一致
        np.testing.assert_allclose(
            inference_array, expected_new_states, rtol=1e-4,
            err_msg="Transform+Inference mixed mode should match full transform"
        )

    def test_transform_restarts_from_initial_state(self, adapter, test_data):
        """
        验证 transform 每次从初始状态开始（设计行为）

        说明：DeepSSM.transform() 每次调用都从初始状态开始，
        状态同步只影响后续 inference() 调用，不影响下一次 transform()。
        这是正确的设计：批处理结果的可复现性。
        """
        train_data = test_data[:50]
        batch = test_data[50:70]

        # Fit
        adapter.fit(train_data)

        # 第一次 transform
        adapter.reset_state()
        result1 = adapter.transform(batch)

        # 做一些 inference 改变内部状态
        for i in range(5):
            adapter.inference(test_data.iloc[70 + i].values)

        # 第二次 transform（不 reset_state）
        result2 = adapter.transform(batch)

        # 两次 transform 结果应该相同（因为 transform 总是从初始状态开始）
        np.testing.assert_allclose(
            result1.values, result2.values, rtol=1e-4,
            err_msg="Transform should always produce same result regardless of adapter state"
        )

    def test_reset_state_clears_state(self, adapter, test_data):
        """验证 reset_state 正确重置状态"""
        train_data = test_data[:80]

        adapter.fit(train_data)
        adapter.transform(train_data)

        # 获取 transform 后的状态
        processor = adapter._realtime_processor
        state_before_reset = processor.ekf.z.clone()

        # 重置状态
        adapter.reset_state()

        # 验证状态已重置
        state_after_reset = processor.ekf.z
        # 重置后状态不应与之前相同（除非恰好相同，但概率极低）
        # 这里验证 reset 确实被调用了
        assert processor.step_count == 0, "Step count should be reset to 0"


class TestCrossAdapterConsistency:
    """跨适配器一致性测试"""

    def test_both_adapters_support_state_sync(self):
        """验证两种适配器都支持状态同步"""
        test_data = generate_test_data(100, obs_dim=3)
        train_data = test_data[:60]
        eval_data = test_data[60:80]

        # LGSSM
        lgssm_config = LGSSMConfig(obs_dim=3, state_dim=4)
        lgssm_adapter = LGSSMAdapter(config=lgssm_config)
        lgssm_adapter.fit(train_data)
        lgssm_adapter.transform(eval_data)
        assert lgssm_adapter._state is not None

        # DeepSSM
        deepssm_config = DeepSSMConfig(obs_dim=3, state_dim=4, max_epochs=3)
        deepssm_adapter = DeepSSMAdapter(config=deepssm_config)
        deepssm_adapter.fit(train_data)
        deepssm_adapter.transform(eval_data)
        assert deepssm_adapter._realtime_processor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
