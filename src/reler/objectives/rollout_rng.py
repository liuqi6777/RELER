"""RELER's optional rank-local action RNG; leaves model/data RNG streams untouched."""

import hashlib
from contextlib import contextmanager

import torch


def validate_rollout_seed(seed):
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
    ):
        raise ValueError("rollout_seed must be an integer in [0, 2**32), or None")


class RolloutRNG:
    """Key draws by seed, rank, step, mode, named stream and within-step call.

    Trainer checkpoints are taken at optimizer boundaries. On resume the next
    global_step starts at call zero, just as in uninterrupted training; no mutable
    device RNG state needs gathering from ZeRO ranks. Train/eval counters are
    separate. Named streams have separate counters; the default action stream
    preserves the original seed keys. Changing world size or accumulation changes this experiment.
    """

    version = 1

    def __init__(self, seed=None):
        self.reset(seed)

    def reset(self, seed):
        validate_rollout_seed(seed)
        self.seed = seed
        self._counters = {}

    @contextmanager
    def draw(self, device, step, training=True, *, stream="rollout"):
        if self.seed is None:
            yield
            return
        device = torch.device(device)
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Independent rollout RNG supports CPU and CUDA only")
        counter_key = (training, stream)
        previous_step, call = self._counters.get(counter_key, (None, 0))
        if previous_step != step:
            call = 0
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        key = (
            f"{stream}-v{self.version}:{self.seed}:{rank}:{step}:{int(training)}:{call}"
        )
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (
            2**63
        )
        devices = []
        if device.type == "cuda":
            devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
        # Beta.sample has no generator argument. Fork only the active device and
        # CPU; do not use torch.manual_seed, which also mutates unrelated GPUs.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(seed)
            yield
        self._counters[counter_key] = (step, call + 1)
