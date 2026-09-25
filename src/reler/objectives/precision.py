"""Keep RELER ranking contractions in FP32 even inside mixed-precision training."""

from contextlib import contextmanager

import torch


@contextmanager
def fp32_scores(device):
    # Autocast disabled alone is insufficient when training enables CUDA TF32:
    # its reduced mantissa can still collapse nearby candidate scores into ties.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device_type=device.type, enabled=False):
            yield
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
