# Jesse-Trade (Custom Jesse Project)

This repository is a production-trading project built on top of the Jesse framework.
By default we install the upstream Jesse package, and we use the `jesse-live` Cython
package for live trading. A patched Jesse submodule exists for experimental fixes.
This README summarizes how the system is installed, started, and how the live strategy
is structured to help with debugging and maintenance.

## Repository layout

- `jesse/` — Jesse submodule (optional patch branch for experiments)
- `strategies/` — live strategies (each folder is independent)
- `src/` — production code (bars/features/indicators/utils)
- `research/` — offline experiments (not imported in production)
- `tests/` — pytest tests

## Install (server)

1. Install dependencies with conda (default: upstream Jesse from PyPI).
   We use a conda environment named `jesse`:

```sh
./install.sh
# or: ./install.sh --dev
```

2. Build/install `jesse-live` (Cython) inside the conda env:

```sh
jesse install-live --no-strict
```

3. (Optional) Use the patched Jesse submodule:

```sh
./install.sh --patch
# or: ./install.sh --dev --patch
```

## Run

`run.sh` is the service entrypoint. It launches `jesse run` from the repository root:

```sh
./run.sh
```

Notes:
- Must be run from repo root so `.env` is loaded.
- `.env` contains database and exchange credentials.

## Live strategy: BinanceBtcDemoBar

Location: `strategies/BinanceBtcDemoBar/`

Data/feature/model flow:
1. Jesse 1m BTC-USDT candles → filter zero-volume candles
2. Build fusion bars via `DemoBar`
3. Compute raw features with `SimpleFeatureCalculator`
4. Per-model feature subset → ARDVAE dimensionality reduction
5. LightGBM models vote; trade only when all agree

Models used:
- `c_L6_N1`
- `r_L6_N3`
- `r_L9_N3`

Execution details:
- Maker limit orders at best bid/ask (orderbook if available)
- Position size = 95% of `leveraged_available_margin`
- Stop-loss based on 5% no-leverage ratio scaled by leverage

## Offline training

- Feature selection: `flow_feature_select.py`
- Model training: `flow_model_build.py`
- Model configs: `strategies/BinanceBtcDemoBar/models/`

## Notes for debugging live issues

- Live trading depends on the `jesse-live` Cython package (installed via `jesse install-live`).
- The `--patch` option installs Jesse from the local submodule (not PyPI).
- The patch branch is used to experiment with stability fixes (e.g., live disconnection
  investigation and resilience improvements). The default setup uses upstream Jesse
  to reproduce issues against the official runtime.
