"""Verify the local machine-learning environment and PyTorch MPS support."""

from __future__ import annotations

import platform
import sys
import time

import lightgbm
import pandas as pd
import sklearn
import torch


def main() -> None:
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        size = 2_048
        left = torch.randn((size, size), device=device)
        right = torch.randn((size, size), device=device)
        torch.mps.synchronize()
        start = time.perf_counter()
        result = left @ right
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1_000
        print(
            f"MPS matrix multiplication: {size}x{size} @ {size}x{size}, "
            f"{elapsed_ms:.2f} ms (checksum: {result[0, 0].item():.6f})"
        )
    else:
        print("MPS matrix multiplication: skipped (MPS is unavailable)")

    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"pandas version: {pd.__version__}")
    print(f"LightGBM version: {lightgbm.__version__}")


if __name__ == "__main__":
    main()
