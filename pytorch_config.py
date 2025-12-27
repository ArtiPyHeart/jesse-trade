"""
Global PyTorch configuration for jesse-trade project.

This module keeps CPU as the default device while allowing explicit CUDA/MPS usage
when a model chooses to accelerate training.
"""

import os
import warnings

# Set environment variables before importing torch
# Allow MPS fallback to CPU for unsupported operators when MPS is used
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import torch after setting environment variables
import torch


def _configure_pytorch():
    """Configure PyTorch defaults while allowing explicit GPU usage."""
    # Force CPU device globally
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")

    # Set default tensor type
    torch.set_default_dtype(torch.float32)

    # Note: torch.set_default_tensor_type is deprecated in PyTorch 2.1+
    # The set_default_device('cpu') above already ensures CPU tensors

    # Log configuration
    device = (
        torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    )
    print(f"PyTorch configured: device={device}, dtype={torch.get_default_dtype()}")


def get_device() -> str:
    """
    Get the configured PyTorch device.

    Returns:
        str: Returns the default device string.
    """
    if hasattr(torch, "get_default_device"):
        return str(torch.get_default_device())
    return "cpu"


def get_torch_dtype(dtype_str: str = "float32") -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Args:
        dtype_str: String representation of dtype ('float32' or 'float64')

    Returns:
        torch.dtype: Corresponding PyTorch dtype
    """
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    else:
        warnings.warn(f"Unsupported dtype '{dtype_str}', defaulting to float32")
        return torch.float32


# Configure PyTorch when this module is imported
_configure_pytorch()
