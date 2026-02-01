# Jesse Watchdog Setup (Live)

This document describes how to run the watchdog that checks:
- `storage/health/candle_heartbeat.json`
- `storage/health/account_heartbeat.json`

and restarts the live session when heartbeats go stale.

## 1) Run directly (foreground)
```bash
cd /root/jesse-trade
/root/miniforge3/envs/jesse/bin/python scripts/jesse_watchdog.py \
  --max-candle-age-seconds 120 \
  --max-account-age-seconds 900 \
  --restart-cmd "/root/jesse-trade/run.sh"
```

## 2) Run once (cron/timer)
```bash
cd /root/jesse-trade
/root/miniforge3/envs/jesse/bin/python scripts/jesse_watchdog.py \
  --once \
  --max-candle-age-seconds 120 \
  --max-account-age-seconds 900 \
  --restart-cmd "/root/jesse-trade/run.sh"
```

## 3) Log file
```bash
/root/miniforge3/envs/jesse/bin/python scripts/jesse_watchdog.py \
  --log-file storage/health/watchdog.log \
  --restart-cmd "/root/jesse-trade/run.sh"
```

Notes:
- If you run jesse via systemd, change `--restart-cmd` to `systemctl restart <service>`.
- `max-account-age-seconds` should be higher than `max-candle-age-seconds` to avoid false restarts.
