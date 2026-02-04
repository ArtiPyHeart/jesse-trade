from pathlib import Path

from src.utils.feature_selection_io import (
    append_selection_row,
    load_existing_selection_keys,
)


def test_append_selection_row_writes_header_and_row(tmp_path: Path) -> None:
    output_path = tmp_path / "selection.csv"
    append_selection_row(
        output_path,
        {
            "log_return_lag": 4,
            "pred_next": 1,
            "label_type": "hard",
            "gmm_random_state": 123,
            "n_total_features": 10,
            "n_selected_features": 3,
            "selected_features": '["a", "b", "c"]',
            "timestamp": "2024-01-01 00:00:00",
        },
    )

    content = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert content[0].startswith("log_return_lag,pred_next,label_type")
    assert len(content) == 2


def test_load_existing_selection_keys(tmp_path: Path) -> None:
    output_path = tmp_path / "selection.csv"
    append_selection_row(
        output_path,
        {
            "log_return_lag": 5,
            "pred_next": 2,
            "label_type": "direction",
        },
    )

    keys = load_existing_selection_keys(output_path)
    assert keys == {(5, 2, "direction")}
