"""
ARD-VAE 设备选择测试

验证训练开始时的 CUDA/MPS/CPU 选择逻辑与回退警告。
"""

import warnings

import torch

from src.features.dimensionality_reduction import ARDVAE, ARDVAEConfig


class TestARDVAEDeviceSelection:
    """训练开始时的设备选择逻辑测试。"""

    @staticmethod
    def _build_model(dtype: str = "float32") -> ARDVAE:
        config = ARDVAEConfig(max_epochs=1, dtype=dtype)
        return ARDVAE(config)

    def test_cuda_preferred_over_mps(self, monkeypatch):
        model = self._build_model()
        monkeypatch.setattr(model, "_cuda_available", lambda: True)
        monkeypatch.setattr(model, "_mps_available", lambda: True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._select_training_device()

        assert model.device == "cuda"
        assert len(caught) == 0

    def test_mps_used_when_no_cuda(self, monkeypatch):
        model = self._build_model()
        monkeypatch.setattr(model, "_cuda_available", lambda: False)
        monkeypatch.setattr(model, "_mps_available", lambda: True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._select_training_device()

        assert model.device == "mps"
        assert len(caught) == 0

    def test_cpu_fallback_warns(self, monkeypatch):
        model = self._build_model()
        monkeypatch.setattr(model, "_cuda_available", lambda: False)
        monkeypatch.setattr(model, "_mps_available", lambda: False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._select_training_device()

        assert model.device == "cpu"
        assert any("回退到 CPU" in str(w.message) for w in caught)

    def test_mps_float64_downgrade(self, monkeypatch):
        model = self._build_model(dtype="float64")
        monkeypatch.setattr(model, "_cuda_available", lambda: False)
        monkeypatch.setattr(model, "_mps_available", lambda: True)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model._select_training_device()

        assert model.device == "mps"
        assert model.config.torch_dtype == torch.float32
        assert any("float64" in str(w.message) for w in caught)
