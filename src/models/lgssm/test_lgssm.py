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

    # Test smoothing
    smoothed_states, smoothed_cov = kf.smooth(filtered_states, covariances, A_t, Q_t)
    assert smoothed_states.shape == (T, state_dim), f"Smoothed states shape mismatch"

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
    states = model.predict(X_train)
    assert states.shape == (
        train_size,
        state_dim,
    ), f"Predicted states shape mismatch: {states.shape}"

    # Test with covariance
    states, cov = model.predict(X_train, return_covariance=True)
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
    batch_states = model.predict(observations[:50])
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
    batch_states_compare = model.predict(observations[50:60])

    # They won't be exactly the same due to different initialization,
    # but shapes should match
    assert (
        realtime_states.shape == batch_states_compare.shape
    ), "Shape mismatch in real-time inference"

    print("✓ Real-time inference tests passed")


def test_predict_update_consistency():
    """Test that predict and update_single produce identical results."""
    print("\nTesting predict vs update_single consistency...")

    # Generate data
    T, obs_dim, state_dim = 100, 10, 5
    observations, _, _ = generate_synthetic_data(T, obs_dim, state_dim)

    # Train model
    config = LGSSMConfig(state_dim=state_dim, max_epochs=20, use_scaler=True, seed=42)
    model = LGSSM(config)
    model.fit(observations[:80], verbose=False)

    # Test data
    test_data = observations[80:100]

    # Method 1: Batch prediction using predict()
    batch_states, batch_covariances = model.predict(test_data, return_covariance=True)

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
        err_msg="States differ between predict() and update_single()",
    )

    np.testing.assert_allclose(
        batch_covariances,
        sequential_covariances,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Covariances differ between predict() and update_single()",
    )

    print("✓ Predict vs update_single consistency tests passed")


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
    original_predictions = model.predict(observations)

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    model.save(tmp_path)

    # Load model
    loaded_model = LGSSM.load(tmp_path)

    # Get predictions from loaded model
    loaded_predictions = loaded_model.predict(observations)

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
    states = model.predict(observations)
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
    states = model.predict(df)

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
    states = model.predict(observations)

    print(f"Generated features shape: {states.shape}")
    print(f"Features sample (first 5 values): {states[-1][:5]}")

    # Test real-time processing
    last_state = states[-2]
    last_cov = np.eye(state_dim) * 0.1
    new_state, new_cov = model.update_single(observations[-1], last_state, last_cov)

    print(f"Real-time new state: {new_state}")

    print("✓ Comparison tests completed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running LGSSM Implementation Tests")
    print("=" * 60)

    test_kalman_filter()
    test_lgssm_basic()
    test_lgssm_realtime()
    test_predict_update_consistency()  # New test for consistency
    test_lgssm_save_load()
    test_lgssm_no_scaler()
    test_lgssm_pandas_input()
    test_parameter_constraints()
    compare_with_original()

    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
