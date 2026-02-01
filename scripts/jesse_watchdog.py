from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    tmp_path.replace(path)


def _extract_last_entry(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(data, dict):
        return None
    if "last" in data and isinstance(data["last"], dict):
        return data["last"]
    if "ts" in data:
        return data
    return None


def _is_stale(
    now_ms: int, entry: Optional[Dict[str, Any]], max_age_seconds: int
) -> bool:
    if not entry or "ts" not in entry:
        return True
    return now_ms - int(entry["ts"]) > max_age_seconds * 1000


def _load_state(state_path: Path) -> Dict[str, Any]:
    data = _read_json(state_path)
    if isinstance(data, dict):
        return data
    return {}


def _save_state(state_path: Path, state: Dict[str, Any]) -> None:
    _atomic_write(state_path, state)


def _log(message: str, log_file: Optional[Path]) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    line = f"[{timestamp}] {message}"
    if log_file is None:
        print(line, flush=True)
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run_restart(restart_cmd: str, log_file: Optional[Path]) -> None:
    try:
        args = shlex.split(restart_cmd)
        subprocess.run(args, check=False)
        _log(f"restart_cmd executed: {restart_cmd}", log_file)
    except Exception as exc:
        _log(f"restart_cmd failed: {exc}", log_file)


def _evaluate(
    health_dir: Path,
    max_candle_age_seconds: int,
    max_account_age_seconds: int,
) -> Tuple[bool, str]:
    now_ms = int(time.time() * 1000)

    candle_path = health_dir / "candle_heartbeat.json"
    account_path = health_dir / "account_heartbeat.json"

    candle_data = _extract_last_entry(_read_json(candle_path))
    account_data = _extract_last_entry(_read_json(account_path))

    candle_stale = _is_stale(now_ms, candle_data, max_candle_age_seconds)
    account_stale = _is_stale(now_ms, account_data, max_account_age_seconds)

    if candle_stale or account_stale:
        detail = []
        if candle_stale:
            detail.append("candle_heartbeat stale")
        if account_stale:
            detail.append("account_heartbeat stale")
        return False, ", ".join(detail)

    return True, "ok"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jesse heartbeat watchdog")
    parser.add_argument("--health-dir", default="storage/health")
    parser.add_argument("--max-candle-age-seconds", type=int, default=120)
    parser.add_argument("--max-account-age-seconds", type=int, default=900)
    parser.add_argument("--check-interval-seconds", type=int, default=10)
    parser.add_argument("--cooldown-seconds", type=int, default=120)
    parser.add_argument("--restart-cmd", default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-file", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    health_dir = Path(args.health_dir)
    log_file = Path(args.log_file) if args.log_file else None
    state_path = health_dir / "watchdog_state.json"

    def check_once() -> None:
        ok, detail = _evaluate(
            health_dir=health_dir,
            max_candle_age_seconds=args.max_candle_age_seconds,
            max_account_age_seconds=args.max_account_age_seconds,
        )

        if ok:
            return

        _log(f"unhealthy: {detail}", log_file)
        if not args.restart_cmd:
            return

        state = _load_state(state_path)
        now = int(time.time())
        last_restart = int(state.get("last_restart_ts", 0))
        if now - last_restart < args.cooldown_seconds:
            _log("restart suppressed due to cooldown", log_file)
            return

        _run_restart(args.restart_cmd, log_file)
        state["last_restart_ts"] = now
        _save_state(state_path, state)

    if args.once:
        check_once()
        return

    while True:
        check_once()
        time.sleep(args.check_interval_seconds)


if __name__ == "__main__":
    main()
