from typing import Optional
import random
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value


class Dropout:
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        assert 0.0 <= p < 1.0
        self.p = p
        self.training: bool = True
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __call__(self, x: Value) -> Value:
        if not self.training or self.p == 0.0:
            return x
        keep = 1 if self.rng.random() >= self.p else 0
        scale = 0.0 if keep == 0 else (1.0 / (1.0 - self.p))
        return x * scale
