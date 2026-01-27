from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values


def load_env_values(env_path: Path) -> dict[str, str]:
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found: {env_path}")

    values = dotenv_values(env_path)
    return {key: value for key, value in values.items() if value is not None}


def get_env_value(
    key: str,
    env_values: dict[str, str],
    default: str | None = None,
    allow_empty: bool = False,
) -> str:
    value = os.environ.get(key)
    if value is None or (value == "" and not allow_empty):
        value = env_values.get(key)

    if value is None or (value == "" and not allow_empty):
        if default is not None:
            return default
        raise ValueError(f"Missing {key} in .env")

    return value


def get_env_date(key: str, env_values: dict[str, str]) -> str:
    value = get_env_value(key, env_values)

    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format for {key}: {value}. Expected YYYY-MM-DD."
        ) from exc

    return value
