from typing import List, Dict
import os
import sys
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value


class Optimizer:
    def __init__(self, params: List[Value]):
        self.params = params
        self.param_names: List[str | None] = [None] * len(params)
        self.weight_decay: float = 0.0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict:
        raise NotImplementedError

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError

    def attach_param_names(self, names: List[str]) -> None:
        if len(names) != len(self.params):
            raise ValueError("attach_param_names length mismatch")
        self.param_names = names


class SGD(Optimizer):
    def __init__(self, params: List[Value], lr: float = 0.01, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self) -> None:
        for i, p in enumerate(self.params):
            name = self.param_names[i]
            decay = self.weight_decay if (name and ".weight." in name) else 0.0
            g = p.grad + decay * p.data
            p.data -= self.lr * g

    def state_dict(self) -> dict:
        return {"type": "SGD", "lr": self.lr, "weight_decay": self.weight_decay}

    def load_state_dict(self, state: dict) -> None:
        if state.get("type") != "SGD":
            raise ValueError("Optimizer state type mismatch: expected SGD")
        self.lr = float(state["lr"])
        self.weight_decay = float(state.get("weight_decay", 0.0))


class SGDMomentum(Optimizer):
    def __init__(self, params: List[Value], lr: float = 0.01, beta: float = 0.9, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.v: Dict[Value, float] = {p: 0.0 for p in self.params}

    def step(self) -> None:
        for i, p in enumerate(self.params):
            name = self.param_names[i]
            decay = self.weight_decay if (name and ".weight." in name) else 0.0
            g = p.grad + decay * p.data
            v_prev = self.v[p]
            v_new = self.beta * v_prev + g
            self.v[p] = v_new
            p.data -= self.lr * v_new

    def state_dict(self) -> dict:
        velocities = [self.v[p] for p in self.params]
        return {"type": "SGDMomentum", "lr": self.lr, "beta": self.beta, "weight_decay": self.weight_decay, "v": velocities}

    def load_state_dict(self, state: dict) -> None:
        if state.get("type") != "SGDMomentum":
            raise ValueError("Optimizer state type mismatch: expected SGDMomentum")
        self.lr = float(state["lr"])
        self.beta = float(state["beta"])
        self.weight_decay = float(state.get("weight_decay", 0.0))
        v_list = state["v"]
        if len(v_list) != len(self.params):
            raise ValueError("Velocity length mismatch in optimizer state")
        for p, v in zip(self.params, v_list):
            self.v[p] = float(v)


class Adam(Optimizer):
    def __init__(
        self,
        params: List[Value],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[Value, float] = {p: 0.0 for p in self.params}
        self.v: Dict[Value, float] = {p: 0.0 for p in self.params}

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            name = self.param_names[i]
            decay = self.weight_decay if (name and ".weight." in name) else 0.0
            g = p.grad + decay * p.data
            m_prev = self.m[p]
            v_prev = self.v[p]
            m = self.beta1 * m_prev + (1.0 - self.beta1) * g
            v = self.beta2 * v_prev + (1.0 - self.beta2) * (g * g)
            self.m[p] = m
            self.v[p] = v
            m_hat = m / (1.0 - self.beta1 ** self.t)
            v_hat = v / (1.0 - self.beta2 ** self.t)
            p.data -= self.lr * (m_hat / (math.sqrt(v_hat) + self.eps))

    def state_dict(self) -> dict:
        m_list = [self.m[p] for p in self.params]
        v_list = [self.v[p] for p in self.params]
        return {
            "type": "Adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "m": m_list,
            "v": v_list,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("type") != "Adam":
            raise ValueError("Optimizer state type mismatch: expected Adam")
        self.lr = float(state["lr"])
        self.beta1 = float(state["beta1"])
        self.beta2 = float(state["beta2"])
        self.eps = float(state["eps"])
        self.weight_decay = float(state.get("weight_decay", 0.0))
        self.t = int(state["t"])
        m_list = state["m"]
        v_list = state["v"]
        if len(m_list) != len(self.params) or len(v_list) != len(self.params):
            raise ValueError("Adam state length mismatch for m/v")
        for p, m, v in zip(self.params, m_list, v_list):
            self.m[p] = float(m)
            self.v[p] = float(v)
