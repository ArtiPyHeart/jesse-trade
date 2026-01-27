from pathlib import Path

import pytest

from src.utils.env_dates import get_env_date, load_env_values


def test_load_env_values_reads_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TRAIN_START_DATE=2022-08-01\nOTHER=foo\n", encoding="utf-8")

    values = load_env_values(env_path)

    assert values["TRAIN_START_DATE"] == "2022-08-01"
    assert values["OTHER"] == "foo"


def test_get_env_date_prefers_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TRAIN_START_DATE=2022-08-01\n", encoding="utf-8")
    values = load_env_values(env_path)

    assert get_env_date("TRAIN_START_DATE", values) == "2022-08-01"

    monkeypatch.setenv("TRAIN_START_DATE", "2022-09-01")
    assert get_env_date("TRAIN_START_DATE", values) == "2022-09-01"


def test_get_env_date_missing_raises(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    values = load_env_values(env_path)

    with pytest.raises(ValueError, match="Missing TRAIN_START_DATE"):
        get_env_date("TRAIN_START_DATE", values)


def test_get_env_date_invalid_format(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("TRAIN_START_DATE=2022/08/01\n", encoding="utf-8")
    values = load_env_values(env_path)

    with pytest.raises(ValueError, match="Invalid date format"):
        get_env_date("TRAIN_START_DATE", values)
