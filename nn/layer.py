from typing import List
from nn.neuron import Neuron
from autograd.value import Value


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "relu"):
        self.neurons: List[Neuron] = [Neuron(n_inputs, activation=activation) for _ in range(n_neurons)]

    def __call__(self, x: List[Value]) -> List[Value]:
        return [n(x) for n in self.neurons]

    def parameters(self) -> List[Value]:
        params: List[Value] = []
        for n in self.neurons:
            params += n.parameters()
        return params
