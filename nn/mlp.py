from typing import List, Optional
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.layer import Layer
from nn.regularization import Dropout


class MLP:
    def __init__(self, n_inputs: int, layer_sizes: List[int], dropout: float = 0.0, dropout_seed: Optional[int] = None):
        assert len(layer_sizes) >= 1
        layers: List[Layer] = []
        in_size = n_inputs
        for i, out_size in enumerate(layer_sizes):
            activation = "identity" if i == len(layer_sizes) - 1 else "relu"
            layers.append(Layer(in_size, out_size, activation=activation))
            in_size = out_size
        self.layers: List[Layer] = layers
        self.training: bool = True
        self.dropouts: List[Dropout] = []
        if dropout and dropout > 0.0:
            for i in range(len(self.layers) - 1):
                seed = None if dropout_seed is None else (dropout_seed + i)
                self.dropouts.append(Dropout(p=dropout, seed=seed))
        else:
            self.dropouts = []

    def __call__(self, x: List[Value]):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1 and self.dropouts:
                do = self.dropouts[i]
                do.training = self.training
                out = [do(v) for v in out]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for layer in self.layers:
            params += layer.parameters()
        return params

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0

    def named_parameters(self) -> List[tuple[str, Value]]:
        names_and_params: List[tuple[str, Value]] = []
        for li, layer in enumerate(self.layers):
            for ni, neuron in enumerate(layer.neurons):
                for wi, w in enumerate(neuron.w):
                    names_and_params.append((f"layers.{li}.neurons.{ni}.weight.{wi}", w))
                names_and_params.append((f"layers.{li}.neurons.{ni}.bias", neuron.b))
        return names_and_params

    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def state_dict(self) -> dict:
        state = {}
        for name, p in self.named_parameters():
            state[name] = p.data
        return state

    def load_state_dict(self, state: dict) -> None:
        model_keys = [name for name, _ in self.named_parameters()]
        state_keys = list(state.keys())
        unknown = [k for k in state_keys if k not in model_keys]
        missing = [k for k in model_keys if k not in state_keys]
        if unknown:
            raise ValueError(f"Unknown parameters in state_dict: {unknown}")
        if missing:
            raise ValueError(f"Missing parameters in state_dict: {missing}")
        for name, p in self.named_parameters():
            val = state[name]
            p.data = float(val)


if __name__ == "__main__":
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    model = MLP(3, [4, 4, 1])
    y = model(x)
    if isinstance(y, list):
        y = y[0]
    y.backward()
    for p in model.parameters():
        assert p.grad != 0
    print("Gradient sanity check passed.")
