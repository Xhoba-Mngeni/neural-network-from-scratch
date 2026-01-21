# Neural Network Library — Pure Python Autograd & MLPs

This repository is a from‑scratch neural network framework implemented in pure Python. It includes a scalar reverse‑mode automatic differentiation engine, trainable MLPs, optimizers, classification utilities, serialization, gradient checking, mini‑batch training, and regularization — all without NumPy or external ML libraries.

## Project Overview
- Builds explicit computation graphs during the forward pass and computes gradients via backpropagation.
- Reverse‑mode autodiff with a Value class for scalar math and operator overloading.
- Educational and engineering focus: clarity and correctness over performance; extensible design for later features.

## Feature Summary
- Scalar autograd engine (Value)
- Neuron → Layer → MLP
- Regression and multi‑class classification
- Softmax and cross‑entropy
- Optimizers: SGD, Momentum, Adam
- Mini‑batching and DataLoader
- Dropout and weight decay
- Model serialization (state_dict)
- Numerical gradient checking

## Architecture Overview
- Core abstraction: Value(data, grad, _prev, _op, _backward)
- Forward: constructs a DAG; each operation returns a new Value with a _backward closure.
- Backward: reverse topological traversal that accumulates gradients via the chain rule.
- Separation of concerns: model (MLP), loss (MSE/BCE/CE), optimizer (SGD/Momentum/Adam), data (Dataset/DataLoader), functional ops (softmax), regularization (Dropout, weight decay), serialization.

```
Computation Graph (example z = (x * y + b).relu())

 x -----*----\
            (mul)----+
 y -----*----/       \
                     (+)----(ReLU)----> z
 b ------------------/
```

```
Backward Pass (reverse topological order)

 z.grad = 1
 ReLU._backward -> (+)._backward -> mul._backward -> x.grad, y.grad, b.grad
```

```
MLP Composition

 [x1, x2, ...] -> Layer(h1) -> Dropout? -> Layer(h2) -> ... -> Layer(out)
```

## Autograd Design
- Reverse‑mode is chosen for efficiency when outputs are few and parameters many.
- Each operation creates a new Value node with pointers to parents (_prev) and a _backward closure that knows local derivatives.
- backward() builds a topological order via DFS, seeds output.grad = 1.0, then calls node._backward() reverse‑ordered once.

```python
from autograd.value import Value
x = Value(2.0)
y = Value(3.0)
b = Value(1.0)
z = (x * y + b).relu()
z.backward()
```

Internally:
- mul produces out = x*y with _backward adding y*out.grad to x.grad and x*out.grad to y.grad
- add propagates out.grad to both inputs
- relu gates the gradient by 1 if output > 0 else 0
- backward walks nodes in reverse topo order and accumulates grads

## Neural Network Design
- Neuron computes y = f(∑ w_i x_i + b) with weights/bias as Value objects and activation relu/identity.
- Layer is a list of Neurons; MLP stacks Layers.
- Activations are externalized: final layer is identity for logits in classification, letting softmax be applied externally.

```python
from nn.mlp import MLP
from autograd.value import Value
model = MLP(3, [4, 4, 1])
y = model([Value(1.0), Value(0.0), Value(-1.0)])
```

## Training Loop & Optimizers
- Loop: forward → loss → zero_grad → backward → optimizer.step
- Zeroing avoids gradient leakage; optimizer abstracts parameter updates and maintains state (velocity/moments).
- SGD vs Momentum vs Adam:
  - SGD: w -= lr * grad
  - Momentum: accumulates velocity v = βv + grad; w -= lr * v
  - Adam: adaptive moments with bias correction; w -= lr * m̂ / (sqrt(v̂) + eps)

```python
from nn.mlp import MLP
from nn.loss import mse_loss
from nn.optim import Adam
from autograd.value import Value

model = MLP(3, [4, 4, 1])
opt = Adam(model.parameters(), lr=0.01)
opt.attach_param_names([name for name, _ in model.named_parameters()])

for epoch in range(100):
    x = [Value(1.0), Value(-2.0), Value(3.0)]
    y_pred = model(x); y_pred = y_pred if not isinstance(y_pred, list) else y_pred[0]
    loss = mse_loss([y_pred], [2.0])
    opt.zero_grad(); loss.backward(); opt.step()
```

## Classification Pipeline
- Model outputs raw logits; softmax turns logits into probabilities; cross‑entropy consumes the target index.
- Separation keeps numerics stable and flexible for future extensions.

```python
from nn.functional import softmax
from nn.loss import cross_entropy
logits = model([Value(1.0), Value(0.5)])
probs = softmax(logits if isinstance(logits, list) else [logits])
loss = cross_entropy(probs, target=2)
```

## Regularization & Generalization
- Dropout: during train(), randomly zero activations with probability p and scale retained ones by 1/(1−p); during eval(), identity.
- Weight decay: handled inside optimizers as L2 penalty added to gradient term (grad + λ·w), applied only to weights (not biases) via parameter name filtering.

## Validation & Correctness
- Gradient checking: numerical finite differences compare against analytical grads from backward(); report max absolute error and fail if above tolerance.
- Ensures autodiff is correct and robust.

## Project Structure
```
autograd/
  value.py             # Scalar autodiff engine (Value: ops, relu, log, exp, backward)
nn/
  neuron.py            # Neuron: weights, bias, activation
  layer.py             # Layer: list of Neurons
  mlp.py               # MLP: stacked Layers, train/eval, dropout hooks, state_dict
  loss.py              # MSE, BCE, cross-entropy
  functional.py        # softmax, argmax, predict, accuracy
  optim.py             # Optimizers: SGD, Momentum, Adam (+ weight decay)
  dataloader.py        # Dataset/DataLoader for mini-batching
  serialization.py     # save_checkpoint/load_checkpoint (JSON)
  gradient_check.py    # numerical_grad and check_gradients utility
  regularization.py    # Dropout module
  train.py             # basic training example (regression/classification)
  train_minibatch.py   # mini-batch training examples
  train_classify.py    # classification training script
  train_regularized.py # regularized training demo
```

## Design Philosophy
- No NumPy: forces explicit clarity of autodiff, graphs, and learning dynamics.
- Scalar‑first: minimal surface area; easier to reason and verify; foundation for tensor support later.
- Clarity > performance: correctness and extensibility prioritized over speed; performance work is future‑facing.
- Tradeoffs: scalar computation is slower; JSON serialization is simple but not compact; optimizers rely on deterministic parameter order and names.

## Future Work
- Tensor support (vectorized operations)
- GPU acceleration
- CNNs / RNNs / attention modules
- Mixed precision and numerical stability tooling
- Performance optimizations (fused ops, graph pruning)

Author: Xhoba Bukho Mngeni 