from typing import List
import random
from autograd.value import Value


class Neuron:
    def __init__(self, n_inputs: int, activation: str = "relu", weight_scale: float = 0.1, bias_init: float = 1.0):
        assert activation in ("relu", "identity")
        self.w: List[Value] = [Value(random.uniform(-1.0, 1.0) * weight_scale) for _ in range(n_inputs)]
        self.b: Value = Value(bias_init)
        self.activation: str = activation

    def __call__(self, x: List[Value]) -> Value:
        assert len(x) == len(self.w)
        out = self.b
        for wi, xi in zip(self.w, x):
            out = out + wi * xi
        return out.relu() if self.activation == "relu" else out

    def parameters(self) -> List[Value]:
        return [*self.w, self.b]
