from typing import List, Union
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value


def mse_loss(y_pred: List[Value], y_true: List[Union[float, Value]]) -> Value:
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    total = Value(0.0)
    for yp, yt in zip(y_pred, y_true):
        yt_val = yt if isinstance(yt, Value) else Value(float(yt))
        diff = yp - yt_val
        total = total + diff ** 2
    return total / n


def bce_loss(y_pred: List[Value], y_true: List[Union[float, Value]], eps: float = 1e-12) -> Value:
    assert len(y_pred) == len(y_true)
    n = len(y_pred)
    total = Value(0.0)
    one = Value(1.0)
    for yp, yt in zip(y_pred, y_true):
        yt_val = yt if isinstance(yt, Value) else Value(float(yt))
        term1 = yt_val * yp.log(eps)
        term2 = (one - yt_val) * (one - yp).log(eps)
        total = total + (term1 + term2)
    return -(total / n)


def cross_entropy(probs: List[Value], target: int, eps: float = 1e-12) -> Value:
    assert 0 <= target < len(probs)
    return -(probs[target].log(eps))
