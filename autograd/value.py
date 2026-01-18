"""
Scalar reverse-mode automatic differentiation engine.
Provides a Value class that builds a computation graph and supports backpropagation.
"""

from __future__ import annotations
from typing import Callable, Iterable, Set
import math


class Value:
    """
    A scalar that participates in a computation graph and supports reverse-mode autodiff.
    """

    def __init__(self, data: float, _children: Iterable[Value] = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: Set[Value] = set(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    # This is the operator overloads
    def __add__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # d(out)/d(self) = 1, d(out)/d(other) = 1
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | Value) -> Value:
        return self.__add__(other)

    def __mul__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        # d(self*other)/d(self) = other, d(self*other)/d(other) = self
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | Value) -> Value:
        return self.__mul__(other)

    def __neg__(self) -> Value:
        out = Value(-self.data, (self,), "neg")

        # d(-self)/d(self) = -1
        def _backward():
            self.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: float | Value) -> Value:
        return self.__add__(-other if isinstance(other, Value) else -Value(other))

    def __rsub__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return other.__sub__(self)

    def __truediv__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # self / other = self * other**(-1)
        return self * other ** -1.0

    def __rtruediv__(self, other: float | Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return other.__truediv__(self)

    def __pow__(self, power: float) -> Value:
        assert isinstance(power, (int, float)), "** expects a scalar exponent"
        out = Value(self.data ** power, (self,), f"**{power}")

        # d(self**p)/d(self) = p * self**(p-1)
        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._backward = _backward
        return out

    # This is the activation function
    def relu(self) -> Value:
        out = Value(self.data if self.data > 0.0 else 0.0, (self,), "ReLU")

        # Gradient flows only if output > 0
        def _backward():
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad

        out._backward = _backward
        return out

    # The Elementary functions
    def log(self, eps: float = 1e-12) -> Value:
        x = self.data if self.data > eps else eps
        out = Value(math.log(x), (self,), "log")

        def _backward():
            self.grad += (1.0 / x) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        x = self.data
        ex = math.exp(x)
        out = Value(ex, (self,), "exp")

        def _backward():
            self.grad += ex * out.grad

        out._backward = _backward
        return out

    # The Backpropagation
    def backward(self) -> None:
        """
        Compute gradients for all nodes in the graph that affect this Value.
        Uses reverse-mode autodiff via a single reverse traversal of a topologically
        ordered DAG of the computation graph.
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed gradient at output
        self.grad = 1.0

        # Traverse in reverse topological order, calling each node's backward once
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    # Sanity tests
    # Test 1: z = (x*y + b).relu()
    x = Value(2.0)
    y = Value(3.0)
    b = Value(1.0)
    z = (x * y + b).relu()
    z.backward()
    print("Test 1 grads:", x.grad, y.grad, b.grad)
    assert abs(x.grad - 3.0) < 1e-12
    assert abs(y.grad - 2.0) < 1e-12
    assert abs(b.grad - 1.0) < 1e-12

    # Reset grads for clarity in second test
    # (fresh graph used below so resets are not strictly necessary)

    # Test 2: d = (a*b)**2
    a = Value(3.0)
    bb = Value(4.0)
    c = a * bb
    d = c ** 2
    d.backward()
    print("Test 2 grads:", a.grad, bb.grad)
    assert abs(a.grad - (2 * (a.data * bb.data) * bb.data)) < 1e-12
    assert abs(bb.grad - (2 * (a.data * bb.data) * a.data)) < 1e-12

    # Optional quick checks for other ops
    u = Value(5.0)
    v = Value(2.0)
    w = u / v
    w.backward()
    print("Division grads:", u.grad, v.grad)
