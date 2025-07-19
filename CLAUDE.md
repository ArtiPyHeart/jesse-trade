# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

jesse-trade is a sophisticated quantitative trading system built on the Jesse framework, designed to handle million-dollar portfolios with stable profitability through rapid iteration and advanced mathematical/physical concepts applied to trading.

## Key Principles

1. **Algorithm Correctness First**: Any calculation error can cause significant financial loss. Always ensure mathematical correctness before optimizing for readability or performance.
2. **Professional Code Quality**: Review all code before committing. Follow best practices and maintain production-ready standards.
3. **Scientific Approach**: Apply advanced mathematical and physical concepts to trading. The team consists of excellent mathematicians and physicists.

## Architecture

### Core Components

- **custom_indicators/**: Custom technical indicators library
  - `prod_indicator/`: Production-ready indicators (adaptive, entropy-based, signal processing)
  - `toolbox/`: Alternative bar construction (dollar bars, entropy bars), feature selection
  - `utils/`: Helper functions including math utilities for trigonometric conversions

- **strategies/**: Trading strategies including ML-driven approaches
- **tests/**: Unit tests for critical components

### Key Technologies

- **Jesse Framework**: Core trading engine
- **NumPy/Pandas**: Numerical computations
- **Numba**: Performance optimization
- **ML Libraries**: CatBoost, LightGBM, scikit-learn
- **Database**: PostgreSQL (for optimization results), Redis (caching)

## Development Commands

```bash
# Install dependencies
chmod +x ./install.sh
./install.sh

# Install dev dependencies
pip install -r requirements-dev.txt
```

## Coding Guidelines

### Python Best Practices

1. **Internal Functions**: Use underscore prefix for internal functions/modules to reduce cognitive load
2. **Numba Optimization**: Always unit test Numba-optimized code to ensure correctness
3. **Indicator Implementation**: Follow the pattern in `custom_indicators/prod_indicator/accumulated_swing_index.py`:
   - Handle `sequential` parameter correctly
   - Return all values when `sequential=True`
   - Return only latest value when `sequential=False`

### Trigonometric Functions

**CRITICAL**: EasyLanguage uses degrees, Python uses radians. Always use `custom_indicators/utils/math.py` for conversions:

```python
from custom_indicators.utils.math import sin_deg, cos_deg, tan_deg, arctan_deg
```

### ML Model Integration

1. Models are stored in `ml_models/` directory
2. Feature engineering follows patterns in `custom_indicators/toolbox/feature_selection.py`
3. Use checkpoint system for long-running experiments

## Project-Specific Patterns

### Custom Bar Types

The project extensively uses alternative bar construction methods:
- Dollar bars (volume-weighted)
- Entropy bars (information-theoretic)
- Implementation in `custom_indicators/toolbox/`

### Indicator Development

When creating new indicators:
1. Place production-ready indicators in `custom_indicators/prod_indicator/`
2. Experimental indicators go in `custom_indicators/experimental_indicator/`
3. Always handle edge cases and validate inputs
4. Use vectorized operations where possible

### Strategy Development

1. Strategies should inherit from Jesse's base strategy
2. Can use custom bar types via `alternative_candles_loader`
3. ML model integration via pickle files in `ml_models/`

## Important Files

- `custom_indicators/utils/math.py`: Trigonometric conversion utilities
- `custom_indicators/prod_indicator/accumulated_swing_index.py`: Reference implementation pattern
- `run_optimization.py`: Main optimization entry point
- `scripts/extract_data_from_jesse_to_csv.py`: Data extraction pipeline
- `optimization/optuna_optimize.py`: Optimization framework

## Database Schema

The project uses PostgreSQL for storing optimization results with tables for:
- Optimization studies
- Trial results
- Parameter combinations
- Performance metrics

## Testing Strategy

1. Unit tests for all custom indicators
2. Backtesting for strategy validation
3. Always verify mathematical correctness
4. Test Numba-optimized code against original implementations