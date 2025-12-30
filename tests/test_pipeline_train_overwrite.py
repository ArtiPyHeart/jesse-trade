import sys
from pathlib import Path

import pytest


pytest.importorskip("lightgbm")
pytest.importorskip("optuna")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pipeline_model_train as pmt  # noqa: E402
import pipeline_model_train_individual as pmi  # noqa: E402


def test_reset_pipeline_dir_removes_existing_dir(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "global_pipeline"
    pipeline_dir.mkdir()
    (pipeline_dir / "dummy.txt").write_text("data")

    pmt._reset_pipeline_dir(pipeline_dir)

    assert not pipeline_dir.exists()


def test_reset_pipeline_dir_individual_removes_existing_dir(tmp_path: Path) -> None:
    pipeline_dir = tmp_path / "c_L1_N1"
    pipeline_dir.mkdir()
    (pipeline_dir / "dummy.txt").write_text("data")

    pmi._reset_pipeline_dir(pipeline_dir, "c_L1_N1")

    assert not pipeline_dir.exists()
