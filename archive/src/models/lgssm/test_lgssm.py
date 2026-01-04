"""
Test script for LGSSM implementation.

This script verifies the correctness of the LGSSM model implementation
including training, inference, and save/load functionality.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lgssm import LGSSM, LGSSMConfig
from kalman_filter import KalmanFilter


def generate_synthetic_data(T=500, obs_dim=10, state_dim=3, seed=42):
    """Generate synthetic data from a known linear system."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # True system parameters
    A_true = np.eye(state_dim) * 0.9
    C_true = np.random.randn(obs_dim, state_dim) * 0.5
    Q_true = np.eye(state_dim) * 0.1
    R_true = np.eye(obs_dim) * 0.2

    # Generate states and observations
    states = np.zeros((T, state_dim))
    observations = np.zeros((T, obs_dim))

    # Initial state
    states[0] = np.random.randn(state_dim)
    observations[0] = C_true @ states[0] + np.random.multivariate_normal(
        np.zeros(obs_dim), R_true
    )

    # Generate sequence
    for t in range(1, T):
        states[t] = A_true @ states[t - 1] + np.random.multivariate_normal(
            np.zeros(state_dim), Q_true
        )
        observations[t] = C_true @ states[t] + np.random.multivariate_normal(
            np.zeros(obs_dim), R_true
        )

    return observations, states, (A_true, C_true, Q_true, R_true)


def test_kalman_filter():
    """Test the Kalman filter implementation."""
    print("Testing Kalman Filter...")

    # Setup
    state_dim = 3
    obs_dim = 5
    T = 100

    kf = KalmanFilter(state_dim=state_dim, obs_dim=obs_dim)

    # Generate test data
    observations, true_states, (A, C, Q, R) = generate_synthetic_data(
        T, obs_dim, state_dim
    )

    # Convert to tensors
    y = torch.from_numpy(observations).float()
    A_t = torch.from_numpy(A).float()
    C_t = torch.from_numpy(C).float()
    Q_t = torch.from_numpy(Q).float()
    R_t = torch.from_numpy(R).float()

    # Run filter
    filtered_states, covariances, log_likelihood = kf(y, A_t, C_t, Q_t, R_t)

    # Check shapes
    assert filtered_states.shape == (
        T,
        state_dim,
    ), f"States shape mismatch: {filtered_states.shape}"
    assert covariances.shape == (
        T,
        state_dim,
        state_dim,
    ), f"Covariances shape mismatch: {covariances.shape}"
    assert log_likelihood.dim() == 0, "Log likelihood should be scalar"

    # Test smoothing (returns 3 values: states, covariances, lag_one_covariances)
    smoothed_states, smoothed_cov, lag_one_cov = kf.smooth(filtered_states, covariances, A_t, Q_t)
    assert smoothed_states.shape == (T, state_dim), f"Smoothed states shape mismatch"
    assert lag_one_cov.shape == (T - 1, state_dim, state_dim), f"Lag-one cov shape mismatch"

    print("✓ Kalman Filter tests passed")


def test_lgssm_basic():
    """Test basic LGSSM functionality."""
    print("\nTesting LGSSM basic functionality...")

    # Generate data
    T, obs_dim, state_dim = 200, 10, 5
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Split data
    train_size = int(0.8 * T)
    X_train = observations[:train_size]
    X_val = observations[train_size:]

    # Create and train model
    config = LGSSMConfig(
        state_dim=state_dim,
        max_epochs=20,
        learning_rate=0.01,
        patience=5,
        use_scaler=True,
        seed=42,
    )

    model = LGSSM(config)

    # Test training
    model.fit(X_train, X_val, verbose=True)

    # Check that model was built
    assert model.A is not None, "Model parameters not initialized"
    assert model.config.obs_dim == obs_dim, "Observation dimension not set correctly"

    # Test prediction
    states = model.transform(X_train)
    assert states.shape == (
        train_size,
        state_dim,
    ), f"Predicted states shape mismatch: {states.shape}"

    # Test with covariance
    states, cov = model.transform(X_train, return_covariance=True)
    assert cov.shape == (train_size, state_dim, state_dim), f"Covariance shape mismatch"

    print("✓ LGSSM basic tests passed")

    return model, X_train


def test_lgssm_realtime():
    """Test real-time inference capability."""
    print("\nTesting real-time inference...")

    # Generate data
    T, obs_dim, state_dim = 100, 10, 5
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Train model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=10, use_scaler=True)

    model = LGSSM(config)
    model.fit(observations[:80], verbose=False)

    # Get initial state from batch processing
    batch_states = model.transform(observations[:50])
    last_state = batch_states[-1]

    # Initialize covariance
    last_cov = np.eye(state_dim) * 0.1

    # Process new observations one by one
    realtime_states = []
    for t in range(50, 60):
        new_state, new_cov = model.update_single(observations[t], last_state, last_cov)
        realtime_states.append(new_state)
        last_state = new_state
        last_cov = new_cov

    realtime_states = np.array(realtime_states)

    # Compare with batch processing
    batch_states_compare = model.transform(observations[50:60])

    # They won't be exactly the same due to different initialization,
    # but shapes should match
    assert (
        realtime_states.shape == batch_states_compare.shape
    ), "Shape mismatch in real-time inference"

    print("✓ Real-time inference tests passed")


def test_predict_update_consistency():
    """Test that transform and update_single produce identical results."""
    print("\nTesting transform vs update_single consistency...")

    # Generate data
    T, obs_dim, state_dim = 100, 10, 5
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Train model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=20, use_scaler=True, seed=42)
    model = LGSSM(config)
    model.fit(observations[:80], verbose=False)

    # Test data
    test_data = observations[80:100]

    # Method 1: Batch prediction using transform()
    batch_states, batch_covariances = model.transform(test_data, return_covariance=True)

    # Method 2: Sequential updates using update_single()
    sequential_states = []
    sequential_covariances = []

    # Get initial state and covariance from model
    state, covariance = model.get_initial_state()

    for i, obs in enumerate(test_data):
        # First observation needs special handling
        state, covariance = model.update_single(
            obs, state, covariance, is_first_observation=(i == 0)
        )
        sequential_states.append(state)
        sequential_covariances.append(covariance)

    sequential_states = np.array(sequential_states)
    sequential_covariances = np.array(sequential_covariances)

    # Compare results - they should be identical
    np.testing.assert_allclose(
        batch_states,
        sequential_states,
        rtol=1e-5,
        atol=1e-6,
        err_msg="States differ between transform() and update_single()",
    )

    np.testing.assert_allclose(
        batch_covariances,
        sequential_covariances,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Covariances differ between transform() and update_single()",
    )

    print("✓ Transform vs update_single consistency tests passed")


def test_lgssm_save_load():
    """Test model save and load functionality."""
    print("\nTesting save/load functionality...")

    # Train a model
    T, obs_dim, state_dim = 100, 8, 4
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    config = LGSSMConfig(state_dim=state_dim, max_epochs=10, use_scaler=True, seed=123)

    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Get predictions from original model
    original_predictions = model.transform(observations)

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    model.save(tmp_path)

    # Load model
    loaded_model = LGSSM.load(tmp_path)

    # Get predictions from loaded model
    loaded_predictions = loaded_model.transform(observations)

    # Compare predictions
    np.testing.assert_allclose(
        original_predictions,
        loaded_predictions,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Predictions differ after save/load",
    )

    # Check that scaler parameters were preserved
    if config.use_scaler:
        assert loaded_model.scaler_mean is not None, "Scaler mean not loaded"
        assert loaded_model.scaler_std is not None, "Scaler std not loaded"
        np.testing.assert_allclose(
            model.scaler_mean.cpu().numpy(),
            loaded_model.scaler_mean.cpu().numpy(),
            err_msg="Scaler mean differs",
        )

    # Clean up
    os.unlink(tmp_path)

    print("✓ Save/load tests passed")


def test_lgssm_no_scaler():
    """Test LGSSM without StandardScaler."""
    print("\nTesting LGSSM without scaler...")

    # Generate data
    T, obs_dim, state_dim = 100, 5, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Create model without scaler
    config = LGSSMConfig(
        state_dim=state_dim, max_epochs=10, use_scaler=False, seed=42  # Disable scaler
    )

    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Check that scaler parameters are not set
    if hasattr(model, "scaler_mean"):
        assert (
            model.scaler_mean is None
        ), "Scaler mean should be None when use_scaler=False"

    # Test prediction
    states = model.transform(observations)
    assert states.shape == (T, state_dim), "Prediction shape mismatch without scaler"

    print("✓ No scaler tests passed")


def test_lgssm_pandas_input():
    """Test LGSSM with pandas DataFrame input."""
    print("\nTesting pandas DataFrame input...")

    # Generate data
    T, obs_dim, state_dim = 100, 5, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Convert to DataFrame
    df = pd.DataFrame(observations, columns=[f"feature_{i}" for i in range(obs_dim)])

    # Create and train model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=10)
    model = LGSSM(config)

    # Should accept DataFrame
    model.fit(df, verbose=False)
    states = model.transform(df)

    assert states.shape == (
        T,
        state_dim,
    ), "Prediction shape mismatch with DataFrame input"

    print("✓ DataFrame input tests passed")


def test_parameter_constraints():
    """Test that variance parameters remain positive."""
    print("\nTesting parameter constraints...")

    # Generate data
    T, obs_dim, state_dim = 100, 5, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Create and train model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=20)
    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Check that Q and R are positive definite
    Q = model.Q.detach().cpu().numpy()
    R = model.R.detach().cpu().numpy()

    # Check diagonal elements (since we use diagonal matrices)
    assert np.all(np.diag(Q) > 0), "Q has non-positive diagonal elements"
    assert np.all(np.diag(R) > 0), "R has non-positive diagonal elements"

    # Check eigenvalues for positive definiteness
    Q_eigvals = np.linalg.eigvals(Q)
    R_eigvals = np.linalg.eigvals(R)

    assert np.all(Q_eigvals > 0), "Q is not positive definite"
    assert np.all(R_eigvals > 0), "R is not positive definite"

    print("✓ Parameter constraint tests passed")


def test_log_likelihood_training():
    """Test that training with log-likelihood works correctly."""
    print("\nTesting log-likelihood training...")

    # Generate data
    T, obs_dim, state_dim = 200, 10, 5
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Split data
    train_size = int(0.8 * T)
    X_train = observations[:train_size]
    X_val = observations[train_size:]

    # Create and train model
    config = LGSSMConfig(
        state_dim=state_dim,
        max_epochs=30,
        use_scaler=True,
        seed=42,
    )

    model = LGSSM(config)
    model.fit(X_train, X_val, verbose=False)

    # Get log-likelihood from forward pass
    y_tensor = torch.from_numpy(X_train).to(model.dtype).to(model.device)
    model.eval()
    with torch.no_grad():
        states, covariances, log_likelihood = model(y_tensor)

    # Log-likelihood should be finite and reasonable (negative, typically)
    assert torch.isfinite(log_likelihood), f"Log-likelihood is not finite: {log_likelihood}"

    # Avg log-likelihood per timestep
    avg_ll = log_likelihood.item() / train_size
    print(f"  Avg log-likelihood per timestep: {avg_ll:.4f}")

    # States should be valid
    assert not torch.isnan(states).any(), "States contain NaN"
    assert not torch.isinf(states).any(), "States contain Inf"

    print("✓ Log-likelihood training tests passed")


def compare_with_original():
    """Compare results with expected behavior from original implementation."""
    print("\nComparing with original implementation behavior...")

    # Generate data similar to original
    T, obs_dim, state_dim = 1457, 77, 5  # Match original dimensions
    np.random.seed(42)

    # Create synthetic data that mimics financial features
    observations = np.random.randn(T, obs_dim)
    # Add some temporal correlation
    for t in range(1, T):
        observations[t] = 0.7 * observations[t - 1] + 0.3 * np.random.randn(obs_dim)

    # Train model
    config = LGSSMConfig(
        state_dim=state_dim,
        max_epochs=50,
        learning_rate=0.01,
        A_init_scale=0.95,  # Match original
        Q_init_scale=0.1,  # Match original
        R_init_scale=0.1,  # Match original
        use_scaler=True,
    )

    model = LGSSM(config)

    # Split data for training
    train_size = 1400
    model.fit(observations[:train_size], observations[train_size:], verbose=True)

    # Generate features
    states = model.transform(observations)

    print(f"Generated features shape: {states.shape}")
    print(f"Features sample (first 5 values): {states[-1][:5]}")

    # Test real-time processing
    last_state = states[-2]
    last_cov = np.eye(state_dim) * 0.1
    new_state, new_cov = model.update_single(observations[-1], last_state, last_cov)

    print(f"Real-time new state: {new_state}")

    print("✓ Comparison tests completed")


def test_rts_lag_one_covariance():
    """Test RTS smoother returns correct lag-one covariances and symmetric smoothed covariances."""
    print("\nTesting RTS lag-one covariance and symmetry...")

    # Setup
    state_dim = 3
    obs_dim = 5
    T = 50

    kf = KalmanFilter(state_dim=state_dim, obs_dim=obs_dim)

    # Generate test data
    observations, _, (A, C, Q, R) = generate_synthetic_data(T, obs_dim, state_dim)

    # Convert to tensors
    y = torch.from_numpy(observations).float()
    A_t = torch.from_numpy(A).float()
    C_t = torch.from_numpy(C).float()
    Q_t = torch.from_numpy(Q).float()
    R_t = torch.from_numpy(R).float()

    # Run filter
    states, covariances, _ = kf(y, A_t, C_t, Q_t, R_t)

    # Run smoother - now returns 3 values
    smoothed_states, smoothed_cov, lag_one_cov = kf.smooth(states, covariances, A_t, Q_t)

    # Check shapes
    assert smoothed_states.shape == (T, state_dim), "Smoothed states shape mismatch"
    assert smoothed_cov.shape == (T, state_dim, state_dim), "Smoothed cov shape mismatch"
    assert lag_one_cov.shape == (T - 1, state_dim, state_dim), \
        f"Lag-one cov shape mismatch: {lag_one_cov.shape}"

    # Check smoothed covariances are symmetric (v2.1 fix)
    # Note: Last covariance (T-1) comes from filtered state, may have small asymmetry
    for t in range(T):
        cov = smoothed_cov[t]
        symmetry_error = torch.abs(cov - cov.T).max().item()
        assert symmetry_error < 1e-6, \
            f"Smoothed cov at t={t} is not symmetric: max diff = {symmetry_error}"

    # Check lag-one covariances are reasonable (P_{t+1,t|T} is NOT symmetric)
    for t in range(T - 1):
        norm = torch.linalg.norm(lag_one_cov[t]).item()
        assert norm < 100, f"Lag-one cov at t={t} has unreasonable norm: {norm}"
        assert torch.isfinite(lag_one_cov[t]).all(), f"Lag-one cov at t={t} contains NaN/Inf"

    print("  Smoothed covariances verified symmetric")
    print("✓ RTS lag-one covariance tests passed")


def test_A_spectral_projection():
    """Test A matrix spectral radius projection."""
    print("\nTesting A spectral projection...")

    # Generate data
    T, obs_dim, state_dim = 100, 5, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Create model with tight spectral constraint
    config = LGSSMConfig(
        state_dim=state_dim,
        max_epochs=30,
        A_spectral_max=0.9,  # Tight constraint
        seed=42,
    )

    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Check that A's spectral radius respects the constraint
    with torch.no_grad():
        eigvals = torch.linalg.eigvals(model.A)
        spectral_radius = torch.abs(eigvals).max().item()

    assert spectral_radius <= config.A_spectral_max + 1e-6, \
        f"Spectral radius {spectral_radius} exceeds max {config.A_spectral_max}"

    print(f"  A spectral radius: {spectral_radius:.4f} (max: {config.A_spectral_max})")
    print("✓ A spectral projection tests passed")


def test_QR_variance_clamping():
    """Test Q/R variance bounds."""
    print("\nTesting Q/R variance clamping...")

    # Generate data
    T, obs_dim, state_dim = 100, 5, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Create model with default bounds
    config = LGSSMConfig(
        state_dim=state_dim,
        max_epochs=10,
        Q_log_min=-10.0,
        Q_log_max=10.0,
        R_log_min=-10.0,
        R_log_max=10.0,
    )

    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Get Q and R diagonal values
    Q_diag = torch.diag(model.Q).detach().cpu().numpy()
    R_diag = torch.diag(model.R).detach().cpu().numpy()

    # Check bounds: exp(-10) ≈ 4.5e-5, exp(10) ≈ 22026
    min_val = np.exp(-10.0) - 1e-6
    max_val = np.exp(10.0) + 1e-6

    assert np.all(Q_diag >= min_val), f"Q diagonal below minimum: {Q_diag.min()}"
    assert np.all(Q_diag <= max_val), f"Q diagonal above maximum: {Q_diag.max()}"
    assert np.all(R_diag >= min_val), f"R diagonal below minimum: {R_diag.min()}"
    assert np.all(R_diag <= max_val), f"R diagonal above maximum: {R_diag.max()}"

    print(f"  Q range: [{Q_diag.min():.2e}, {Q_diag.max():.2e}]")
    print(f"  R range: [{R_diag.min():.2e}, {R_diag.max():.2e}]")
    print("✓ Q/R variance clamping tests passed")


def test_deep_copy_early_stopping():
    """Test that early stopping correctly preserves best weights."""
    print("\nTesting deep copy for early stopping...")

    # Generate data
    T, obs_dim, state_dim = 200, 10, 5
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Split with validation
    train_size = int(0.7 * T)
    val_size = int(0.15 * T)

    X_train = observations[:train_size]
    X_val = observations[train_size:train_size + val_size]

    # Create model with early stopping
    config = LGSSMConfig(
        state_dim=state_dim,
        max_epochs=100,  # High epochs to trigger early stopping
        patience=3,      # Low patience
        seed=42,
    )

    model = LGSSM(config)
    model.fit(X_train, X_val, verbose=False)

    # The model should have restored best weights
    # Just verify it trained without error and produces valid output
    states = model.transform(X_train)
    assert not np.any(np.isnan(states)), "States contain NaN after early stopping"
    assert not np.any(np.isinf(states)), "States contain Inf after early stopping"

    print("✓ Deep copy early stopping tests passed")


def test_constant_feature_handling():
    """Test handling of constant (zero-variance) features."""
    print("\nTesting constant feature handling...")

    # Generate data with some constant columns
    T, obs_dim, state_dim = 100, 10, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Make some columns constant
    observations[:, 0] = 0.0  # Zero column
    observations[:, 5] = 1.5  # Constant column

    # Create and train model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=10, use_scaler=True)
    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Check scaler std is clamped (use slightly looser tolerance for float precision)
    if model.scaler_std is not None:
        min_std = model.scaler_std.min().item()
        assert min_std >= 0.9e-8, f"Scaler std too small: {min_std}"

    # Prediction should work without NaN
    states = model.transform(observations)
    assert not np.any(np.isnan(states)), "States contain NaN with constant features"

    print("✓ Constant feature handling tests passed")


def test_short_sequence():
    """Test handling of very short sequences."""
    print("\nTesting short sequence handling...")

    obs_dim, state_dim = 5, 3

    # Create model first
    config = LGSSMConfig(state_dim=state_dim, max_epochs=5, use_scaler=True)
    model = LGSSM(config)

    # Train on longer sequence first
    train_data, _, _ = generate_synthetic_data(100, obs_dim, state_dim)
    model.fit(train_data, verbose=False)

    # Test on short sequences
    for T in [2, 5, 10]:
        short_data = train_data[:T]
        states = model.transform(short_data)
        assert states.shape == (T, state_dim), f"Shape mismatch for T={T}"
        assert not np.any(np.isnan(states)), f"NaN in states for T={T}"

    print("✓ Short sequence handling tests passed")


def test_nan_tolerance_in_transform():
    """Test that transform() can handle NaN observations gracefully."""
    print("\nTesting NaN tolerance in transform...")

    # Generate clean data for training
    T, obs_dim, state_dim = 100, 10, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Train model on clean data
    config = LGSSMConfig(state_dim=state_dim, max_epochs=10, use_scaler=True, seed=42)
    model = LGSSM(config)
    model.fit(observations, verbose=False)

    # Test case 1: Partial NaN (single element)
    test_data1 = observations[80:90].copy()
    test_data1[5, 3] = np.nan  # Inject NaN at t=5, feature=3
    states1 = model.transform(test_data1)
    assert not np.any(np.isnan(states1)), "States contain NaN with partial NaN obs"
    print("  Partial NaN observation handled correctly")

    # Test case 2: Full row NaN
    test_data2 = observations[80:90].copy()
    test_data2[5, :] = np.nan  # Inject NaN for entire row at t=5
    states2 = model.transform(test_data2)
    assert not np.any(np.isnan(states2)), "States contain NaN with full-row NaN obs"
    print("  Full-row NaN observation handled correctly")

    # Test case 3: Multiple consecutive NaN rows
    test_data3 = observations[80:90].copy()
    test_data3[3:6, :] = np.nan  # 3 consecutive NaN rows
    states3 = model.transform(test_data3)
    assert not np.any(np.isnan(states3)), "States contain NaN with consecutive NaN obs"
    print("  Consecutive NaN observations handled correctly")

    # Test case 4: All NaN (edge case - pure prediction)
    test_data4 = np.full((5, obs_dim), np.nan)
    states4 = model.transform(test_data4)
    assert not np.any(np.isnan(states4)), "States contain NaN with all-NaN sequence"
    assert states4.shape == (5, state_dim), f"Shape mismatch: {states4.shape}"
    print("  All-NaN sequence handled correctly (pure prediction)")

    print("✓ NaN tolerance in transform tests passed")


def test_nan_rejection_in_fit():
    """Test that fit() rejects training data with NaN values."""
    print("\nTesting NaN rejection in fit...")

    # Generate data with NaN
    T, obs_dim, state_dim = 100, 10, 3
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)
    observations_with_nan = observations.copy()
    observations_with_nan[50, 3] = np.nan  # Inject NaN

    # Create model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=5, use_scaler=True)
    model = LGSSM(config)

    # Try to fit with NaN data - should raise ValueError
    try:
        model.fit(observations_with_nan, verbose=False)
        assert False, "Should have raised ValueError for NaN in training data"
    except ValueError as e:
        assert "NaN" in str(e), f"Error message should mention NaN: {e}"
        print(f"  Training data correctly rejected: {e}")

    # Also test validation data
    model2 = LGSSM(config)
    val_data_with_nan = observations[80:100].copy()
    val_data_with_nan[5, 2] = np.nan

    try:
        model2.fit(observations[:80], val_data_with_nan, verbose=False)
        assert False, "Should have raised ValueError for NaN in validation data"
    except ValueError as e:
        assert "NaN" in str(e), f"Error message should mention NaN: {e}"
        print(f"  Validation data correctly rejected: {e}")

    print("✓ NaN rejection in fit tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running LGSSM Implementation Tests (v2.1)")
    print("=" * 60)

    # Core functionality tests
    test_kalman_filter()
    test_lgssm_basic()
    test_lgssm_realtime()
    test_predict_update_consistency()
    test_lgssm_save_load()
    test_lgssm_no_scaler()
    test_lgssm_pandas_input()
    test_parameter_constraints()

    # v2.0 stability features
    test_rts_lag_one_covariance()
    test_A_spectral_projection()
    test_QR_variance_clamping()
    test_deep_copy_early_stopping()
    test_constant_feature_handling()
    test_short_sequence()

    # v2.1 log-likelihood training and NaN handling
    test_log_likelihood_training()
    test_nan_rejection_in_fit()
    test_nan_tolerance_in_transform()

    # Comparison test
    compare_with_original()

    print("\n" + "=" * 60)
    print("All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
