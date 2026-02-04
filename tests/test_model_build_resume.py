from pathlib import Path

from src.utils.model_build_resume import model_artifacts_exist


def test_model_artifacts_exist_requires_all_files(tmp_path: Path) -> None:
    model_dir = tmp_path / "c_L4_N1"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = "c_L4_N1"

    assert not model_artifacts_exist(model_dir, model_name)

    required = [
        "features.json",
        "tuning_result.json",
        f"model_{model_name}.txt",
        f"{model_name}.safetensors",
        f"{model_name}.json",
    ]
    for filename in required[:-1]:
        (model_dir / filename).write_text("x", encoding="utf-8")
        assert not model_artifacts_exist(model_dir, model_name)

    (model_dir / required[-1]).write_text("x", encoding="utf-8")
    assert model_artifacts_exist(model_dir, model_name)
