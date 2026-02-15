"""Device configuration for JAX across CPU, MPS (Apple Silicon), and CUDA."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

# Configure JAX memory settings before importing jax
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

if TYPE_CHECKING:
    from jax.sharding import Mesh


def setup_device(prefer: str | None = None) -> jax.Device:  # pyright: ignore[reportInvalidTypeForm]
    """
    Auto-detect and configure JAX device (CPU, MPS, or CUDA).

    Args:
        prefer: Optional platform preference ('cpu', 'mps', 'gpu'). If None, auto-selects best available.

    Returns:
        The configured JAX device.

    Environment variables:
        JAX_PLATFORM: Force specific platform (cpu/mps/gpu)
    """
    env_platform = os.getenv("JAX_PLATFORM")
    if env_platform:
        prefer = env_platform.lower()

    available = jax.devices()
    available_types = {d.platform for d in available}

    platform_map = {"mps": "METAL", "gpu": "gpu", "cuda": "gpu", "cpu": "cpu"}

    if prefer:
        prefer_normalized = platform_map.get(prefer.lower(), prefer.upper())
        matching = [d for d in available if d.platform.upper() == prefer_normalized.upper()]
        if matching:
            device = matching[0]
            print(f"Using {device.platform} device: {device}")
            return device
        else:
            warnings.warn(
                f"Requested platform '{prefer}' not available. "
                f"Available: {available_types}. Falling back to auto-select.",
                stacklevel=2,
            )

    for platform in ["gpu", "METAL", "cpu"]:
        matching = [d for d in available if d.platform == platform]
        if matching:
            device = matching[0]
            print(f"Auto-selected {device.platform} device: {device}")
            return device

    device = available[0]
    print(f"Using default device: {device}")
    return device


def get_device_info() -> dict[str, str | int | list[str]]:
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
    device_count = info["device_count"]
    assert isinstance(device_count, int)
    if device_count > 1:
        all_devices = info["all_devices"]
        assert isinstance(all_devices, list)
        print("All devices:")
        for d in all_devices:
            print(f"  - {d}")
    print("================================\n")


def setup_mesh(num_devices: int | None = None) -> Mesh:
    """
    Create a mesh for tensor parallelism across devices.

    Uses jax.make_mesh which creates an Explicit-mode mesh, enabling
    sharding annotations on nnx.Param (sharding=('tp', None, None)).

    With a single device, expert sharding annotations become no-ops.
    With multiple devices, experts are sharded along the 'tp' axis.

    Args:
        num_devices: Number of devices to use. If None, uses all available.

    Returns:
        JAX Mesh with 'tp' axis.

    Environment variables:
        JAX_TP_DEVICES: Override number of devices for tensor parallelism.
    """
    if num_devices is None:
        num_devices = int(os.getenv("JAX_TP_DEVICES", len(jax.devices())))

    available = jax.devices()
    if num_devices > len(available):
        raise ValueError(f"Requested {num_devices} devices but only {len(available)} available")

    mesh = jax.make_mesh((num_devices,), ("tp",), devices=available[:num_devices])

    print(f"Mesh: {num_devices} device(s), axis='tp'")
    return mesh


# Auto-setup on import (can be disabled with JAX_PLATFORM=none)
if os.getenv("JAX_PLATFORM", "").lower() != "none":
    _default_device = setup_device()
