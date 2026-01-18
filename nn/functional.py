from typing import List, Tuple
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value


def softmax(logits: List[Value]) -> List[Value]:
    assert len(logits) > 0
    shift = max(l.data for l in logits)
    exps = [(l - shift).exp() for l in logits]
    total = Value(0.0)
    for e in exps:
        total = total + e
    return [e / total for e in exps]


def argmax(probs: List[Value]) -> int:
    best_i = 0
    best_v = probs[0].data
    for i in range(1, len(probs)):
        if probs[i].data > best_v:
            best_v = probs[i].data
            best_i = i
    return best_i


def predict(probs: List[Value]) -> int:
    return argmax(probs)


def accuracy(model, dataset: List[Tuple[List[float], int]]) -> float:
    correct = 0
    for x_vals, y in dataset:
        x = [Value(v) for v in x_vals]
        logits = model(x)
        probs = softmax(logits if isinstance(logits, list) else [logits])
        if predict(probs) == y:
            correct += 1
    return correct / len(dataset)
