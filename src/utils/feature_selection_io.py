from __future__ import annotations

import csv
from pathlib import Path


SELECTION_FIELDNAMES = [
    "log_return_lag",
    "pred_next",
    "label_type",
    "gmm_random_state",
    "n_total_features",
    "n_selected_features",
    "selected_features",
    "timestamp",
]


def load_existing_selection_keys(output_path: Path) -> set[tuple[int, int, str]]:
    if not output_path.exists():
        return set()

    keys: set[tuple[int, int, str]] = set()
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                log_return_lag = int(row["log_return_lag"])
                pred_next = int(row["pred_next"])
                label_type = str(row["label_type"])
            except (KeyError, TypeError, ValueError):
                continue
            keys.add((log_return_lag, pred_next, label_type))
    return keys


def append_selection_row(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    row_out = {key: row.get(key, "") for key in SELECTION_FIELDNAMES}

    with output_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SELECTION_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row_out)
        f.flush()
