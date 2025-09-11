# Jesse Trade Project Knowledge Base

## Project Overview
**Jesse Trade** is a quantitative trading system built on the Jesse framework, focusing on algorithmic trading with advanced mathematical and physical concepts applied to financial markets.

## Core Principles
- **Algorithm Correctness First**: Calculation errors can cause significant financial losses. Always prioritize correctness over optimization.
- **Production Standards**: Maintain production-ready code quality. All commits must be reviewed.
- **Scientific Method**: Apply advanced mathematics/physics concepts to trading strategies.
- **Risk Management**: Every strategy implementation must consider risk management as a primary concern.

## Critical Safety Rules
- **Never import from `research/` or `extern/` in production code**
- **All production imports must come from `src/`**
- **Strategies must remain independent - avoid cross-dependencies**
- **Always validate inputs with `assert` statements**
- **Never use broad `try/except` blocks that mask errors**

## Project Structure

### Core Modules
- `src/` - Production code (ONLY source for production imports)
  - `bars/` - Custom bar types (Dollar/Range/Entropy Bars, DEAP symbolic regression)
  - `features/` - Feature engineering and selection
  - `indicators/` - Technical indicators
    - `prod/` - Stable, production-ready indicators
    - `experimental/` - Experimental indicators under development
  - `utils/` - Mathematical tools and utilities
  - `data_process/` - Data transformation and preprocessing

### Strategy & Research
- `strategies/` - Jesse trading strategies (each strategy in separate directory)
- `research/` - Offline experiments and notebooks (NEVER import in production)
- `archive/` - Historical experiments and deprecated code

### External Resources
- `extern/` - Reference materials and third-party code (DO NOT import)
  - Contains trading literature and reference implementations
  - Includes "Trading Systems and Methods" resources

### Infrastructure
- `docker/` - Docker stack (Jesse, PostgreSQL, Redis, pgbouncer)
- `tests/` - pytest unit tests
- `pgbouncer/` - Connection pooling configuration

## Development Environment

### Setup
```bash
# Create environment
conda create -n jesse python=3.11 -y
conda activate jesse

# Install dependencies
./install.sh                      # Production dependencies
pip install -r requirements-dev.txt  # Development dependencies

# Run services
./run.sh  # Starts Jesse services
```

### Docker Services
- Jesse application
- PostgreSQL database
- Redis cache
- pgbouncer connection pooler

## Jesse Candle Format
**Standard Format**: 6-column NumPy array
```
[timestamp, open, close, high, low, volume]
```
- Conversion utility: `numpy_candles_to_dataframe(candles)`
- Custom bar types: Dollar Bar, Range Bar, Entropy Bar
- Advanced: DEAP-based symbolic regression for bar fusion

## Indicator Development Standards

### Location Rules
- Stable indicators → `src/indicators/prod/`
- Experimental → `src/indicators/experimental/`
- Class-based indicators inherit from `_cls_ind.py` base class

### Return Conventions
- `sequential=True`: Returns full time series
- `sequential=False`: Returns latest/final value only
- Length consistency: Use `np.nan` padding to match candle array length

### Implementation Guidelines
- Vectorize operations using NumPy/Numba
- Use `src/utils/math_tools.py` for mathematical operations
- Convert EasyLanguage degrees to Python radians
- Prioritize performance after correctness

## Feature Engineering

### SimpleFeatureCalculator
- Central feature computation system
- Located in `src/features/simple_feature_calculator/`
- Supports batch and sequential computation
- Features must maintain consistent return format

### Feature Selection
- FCQ-based selectors (Feature Clustering Quality)
- Random Forest and CatBoost variants
- Located in `src/features/feature_selection/`

## Testing Standards

### Test Organization
- Unit tests in `tests/` and `src/**/tests/`
- Naming: `test_*.py` for pytest discovery
- Direct-run tests avoid `test_` prefix

### Test Coverage Requirements
- All indicators must test `sequential=True/False` behavior
- Bar construction edge cases
- Feature consistency validation
- Strategy backtesting results

### Running Tests
```bash
pytest -q                          # Run all tests
pytest tests/test_merge.py        # Specific file
pytest -k "test_specific_name"    # Specific test
```

## Coding Standards

### Style Guide
- Follow PEP-8 with 4-space indentation
- Type annotations required for public functions
- Docstrings for all public APIs
- Internal functions prefixed with `_`

### Naming Conventions
- Modules/variables: `snake_case`
- Classes: `CamelCase`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `test_*.py`

### Error Handling
- Use `assert` for input validation
- Fail fast on invalid inputs
- Avoid broad exception catching
- Log errors with context

## Mathematical Utilities

### Key Functions (src/utils/math_tools.py)
- Trigonometric: `deg_sin`, `deg_cos`, `deg_tan` (EasyLanguage compatibility)
- Time series: `dt` (difference), `lag`, `std`
- Statistical: Various statistical operations

## Strategy Development

### Structure
- Each strategy in separate directory under `strategies/`
- Contains: `__init__.py`, `config.py`, model files, features
- Independent - no cross-strategy dependencies

### Key Strategies
- `BinanceBtcEntropyBarV1` - Entropy bar-based BTC strategy
- `BinanceBtcDBar5hAllOrNothing` - Dollar bar strategy
- `MLV1Partial` - Machine learning v1 with partial positions

## Research Tools

### Notebooks
- Label generation: `1. labels.ipynb`
- Feature engineering: `2. features.ipynb`
- Feature selection: `3. feature_selection.ipynb`
- Model training: `4. models.ipynb`

### Research Modules
- `research/optuna_config.py` - Hyperparameter optimization
- `research/bar_research/` - Custom bar type research
- `research/labeler/` - ZigZag and GMM labeling methods

## Production Deployment

### Configuration
- Secrets in `.env` file (never commit)
- Reference `.env.example` for structure
- Production models in strategy directories
- Feature metadata in `feature_info.json`

### Performance Optimization
- Use NumPy vectorization
- Numba JIT compilation where appropriate
- Profile before optimizing
- Memory management for large datasets

## Version Control

### Commit Messages
- Imperative mood for subject line
- Optional prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`
- Reference issues when applicable

### Pull Request Guidelines
- Describe changes and motivation
- Include test coverage
- Add visualizations to `outputs/` if relevant
- Ensure all tests pass locally

## Common Pitfalls to Avoid

1. **Never import from research/ in production**
2. **Don't modify docker-compose.yml without understanding impacts**
3. **Avoid changing pgbouncer settings carelessly**
4. **Don't use relative imports across module boundaries**
5. **Never store secrets in code**
6. **Don't skip unit tests for new features**
7. **Avoid non-vectorized operations in indicators**
8. **Don't mix sequential=True/False return formats**

## Useful Commands

```bash
# Start Jesse dashboard
jesse run

# Run specific backtest
jesse backtest 2021-01-01 2021-12-31

# Generate candles
jesse import-candles "Binance" "BTC-USDT" 2021-01-01

# Run optimization
jesse optimize

# Check strategy performance
jesse report
```

## Notes for AI Assistants

- When modifying indicators, ensure both `sequential=True` and `sequential=False` work correctly
- Always check if similar functionality exists before creating new features
- Maintain backward compatibility when updating production indicators
- Test with real market data, not just synthetic data
- Consider computational efficiency for real-time trading
- Document any non-obvious mathematical transformations
- Preserve existing test coverage when refactoring

## Recent Updates

- Added numpy_fracdiff indicator for fractional differentiation
- Enhanced SimpleFeatureCalculator with entropy features
- Fixed sequential consistency issues across all features
- Improved feature name standardization
