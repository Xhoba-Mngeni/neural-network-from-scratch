# Architecture Overview

This framework is organized around clear, minimal abstractions. The Value class builds computation graphs; higher‑level modules assemble trainable networks; utilities handle training, evaluation, and persistence.

## Modules and Responsibilities
- autograd/value.py: scalar reverse‑mode autodiff
- nn/neuron.py: single neuron (weights, bias, activation)
- nn/layer.py: list of neurons
- nn/mlp.py: stacked layers, dropout, train/eval, named/state dict
- nn/loss.py: MSE, BCE, cross‑entropy
- nn/functional.py: softmax, argmax, predict, accuracy
- nn/optim.py: SGD, Momentum, Adam (+ weight decay)
- nn/dataloader.py: Dataset and DataLoader for mini‑batches
- nn/serialization.py: save/load checkpoints (JSON)
- nn/gradient_check.py: numerical gradient verification
- nn/regularization.py: Dropout module

## Data Flow
```text
Inputs (Values) --> Model (MLP) --> Logits/Outputs
                      |              |
                      v              v
                  Softmax (cls)     Loss (MSE/CE)
                      |              |
                      +-------> backward()
                                    |
                                    v
                             Optimizer.step()
```

## Computation Graph and Backprop
```text
Forward DAG:
  nodes connect by operations
  each node holds parents and _backward closure

Backward traversal:
  reverse topological order
  _backward accumulates grads into parents
```

## MLP Composition
```text
Input -> Layer(h1) -> Dropout? -> Layer(h2) -> ... -> Layer(out)
```

## Separation of Concerns
- Model: defines structure and forward computation
- Loss: turns predictions + targets into a scalar Value
- Optimizer: updates parameters and manages state
- Data: iteration, batching, shuffling
- Regularization: dropout; weight decay in optimizer
- Serialization: explicit, human‑readable state dicts

