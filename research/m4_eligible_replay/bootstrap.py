"""Expose the hash-bound bundled Megatron sources before standalone imports."""

from pathlib import Path
import sys


def initialize_bundled_megatron() -> Path:
    """Configure the same bundled core used by NeMo-RL initialization."""
    root = Path(__file__).resolve().parents[2]
    core = (
        root / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
    )
    if not (core / "megatron/core/__init__.py").is_file():
        raise FileNotFoundError(f"qualified bundled Megatron source missing: {core}")
    if str(core) not in sys.path:
        sys.path.insert(0, str(core))
    return core
