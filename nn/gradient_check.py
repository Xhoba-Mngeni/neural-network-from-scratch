from typing import List, Callable, Union
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.mlp import MLP
from nn.loss import mse_loss


def numerical_grad(f: Callable[[], Value], params: List[Value], eps: float = 1e-6) -> List[float]:
    grads: List[float] = []
    for p in params:
        orig = p.data
        p.data = orig + eps
        fp = f().data
        p.data = orig - eps
        fm = f().data
        p.data = orig
        grads.append((fp - fm) / (2.0 * eps))
    return grads


def check_gradients(
    model,
    loss_fn: Callable[[Union[Value, List[Value]], Union[float, List[float], int]], Value],
    x: List[Value],
    y: Union[float, List[float], int],
    tol: float = 1e-4,
    eps: float = 1e-6,
    verbose: bool = True,
) -> None:
    def f() -> Value:
        out = model(x)
        return loss_fn(out, y)

    model.zero_grad()
    loss = f()
    loss.backward()
    analytical = [p.grad for p in model.parameters()]
    numerical = numerical_grad(f, model.parameters(), eps=eps)
    diffs = [abs(a - n) for a, n in zip(analytical, numerical)]
    max_diff = max(diffs) if diffs else 0.0
    if verbose:
        print("Max gradient diff:", max_diff)
    if max_diff > tol:
        raise AssertionError(f"Gradient check failed: max diff {max_diff} > tol {tol}")


if __name__ == "__main__":
    model = MLP(2, [4, 4, 1])
    x = [Value(1.0), Value(-2.0)]
    y_true = [2.0]

    def loss_fn(pred, target):
        if isinstance(pred, list):
            pred = pred[0]
        return mse_loss([pred], target)

    check_gradients(model, loss_fn, x, y_true)
    print("Gradient check passed.")
