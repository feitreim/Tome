"""Device configuration for JAX across CPU, MPS (Apple Silicon), and CUDA."""

import os
import warnings
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from jax import Device


def setup_device(prefer: str | None = None) -> "Device":
    """
    Auto-detect and configure JAX device (CPU, MPS, or CUDA).

    Args:
        prefer: Optional platform preference ('cpu', 'mps', 'gpu'). If None, auto-selects best available.

    Returns:
        The configured JAX device.

    Environment variables:
        JAX_PLATFORM: Force specific platform (cpu/mps/gpu)
    """
    # Check environment override
    env_platform = os.getenv("JAX_PLATFORM")
    if env_platform:
        prefer = env_platform.lower()

    available = jax.devices()
    available_types = {d.platform for d in available}

    # Map prefer to JAX platform names
    platform_map = {"mps": "METAL", "gpu": "gpu", "cuda": "gpu", "cpu": "cpu"}

    if prefer:
        prefer_normalized = platform_map.get(prefer.lower(), prefer.upper())
        # Try to find matching device
        matching = [d for d in available if d.platform.upper() == prefer_normalized.upper()]
        if matching:
            device = matching[0]
            print(f"Using {device.platform} device: {device}")
            return device
        else:
            warnings.warn(f"Requested platform '{prefer}' not available. Available: {available_types}. Falling back to auto-select.")

    # Auto-select: prefer GPU > MPS > CPU
    for platform in ["gpu", "METAL", "cpu"]:
        matching = [d for d in available if d.platform == platform]
        if matching:
            device = matching[0]
            print(f"Auto-selected {device.platform} device: {device}")
            return device

    # Fallback to first available
    device = available[0]
    print(f"Using default device: {device}")
    return device


def get_device_info() -> dict[str, str | int]:
    """Get information about current JAX device configuration."""
    devices = jax.devices()
    default_device = jax.devices()[0]

    return {
        "default_backend": jax.default_backend(),
        "default_device": str(default_device),
        "platform": default_device.platform,
        "device_count": len(devices),
        "all_devices": [str(d) for d in devices],
    }


def print_device_info() -> None:
    """Print current JAX device configuration."""
    info = get_device_info()
    print("\n=== JAX Device Configuration ===")
    print(f"Backend: {info['default_backend']}")
    print(f"Platform: {info['platform']}")
    print(f"Default device: {info['default_device']}")
    print(f"Total devices: {info['device_count']}")
    if info["device_count"] > 1:
        print("All devices:")
        for d in info["all_devices"]:
            print(f"  - {d}")
    print("================================\n")


# Auto-setup on import (can be disabled with JAX_PLATFORM=none)
if os.getenv("JAX_PLATFORM", "").lower() != "none":
    _default_device = setup_device()
