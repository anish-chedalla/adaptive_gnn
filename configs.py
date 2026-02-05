from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import random

import numpy as np

try:
    import torch
except Exception as e:
    torch = None  # allows importing configs without torch installed


def set_global_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Reproducibility:
    - python RNG
    - numpy RNG
    - torch RNG (cpu/cuda)
    """
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            # Determinism is worth it for debugging.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Some ops still nondeterministic; this enforces stricter behavior.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


@dataclass(frozen=True)
class DataConfig:
    window_W: int = 5
    mode: str = "selfsup"  # "selfsup" or "finetune"
    require_circuit_for_eval: bool = True


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    edge_emb_dim: int = 16
    num_layers: int = 4
    dropout: float = 0.1
    # TemporalConv params
    temporal_channels: int = 32
    temporal_kernel: int = 3


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 1234
    device: str = "cuda"  # "cuda" or "cpu"
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    log_every: int = 50


@dataclass(frozen=True)
class BPConfig:
    max_iters: int = 30
    damping: float = 0.0
    llr_clip: float = 20.0
    eps: float = 1e-9


@dataclass(frozen=True)
class EvalConfig:
    uniform_prior: float = 0.05
    # optional fixed reweight baseline (provided at eval time; NOT stored in dataset)
    p_baseline: Optional[float] = None
    eta_baseline: Optional[float] = None
