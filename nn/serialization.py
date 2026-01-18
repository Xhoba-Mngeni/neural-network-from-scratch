import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Tuple
from nn.mlp import MLP
from nn.optim import Optimizer


def save_checkpoint(model: MLP, optimizer: Optimizer, path: str) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def load_checkpoint(model: MLP, optimizer: Optimizer, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    model_state = state.get("model")
    opt_state = state.get("optimizer")
    if model_state is None or opt_state is None:
        raise ValueError("Checkpoint missing 'model' or 'optimizer' state")
    model.load_state_dict(model_state)
    optimizer.load_state_dict(opt_state)
