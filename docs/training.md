# Training & Regularization

This document summarizes training flows, losses, optimizers, mini‑batches, and regularization.

## Core Training Loop
```python
loss = loss_fn(model(x), y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
- Zeroing avoids gradient accumulation across steps.
- backward() computes analytical grads via reverse‑mode autodiff.
- Optimizer updates parameters; Momentum/Adam maintain internal state.

## Losses
- MSE: mean of squared errors for regression
- BCE: binary cross‑entropy (assumes probabilities in (0,1))
- Cross‑entropy: L = −log(p_y) for classification; softmax outputs probabilities

## Classification Stack
```python
logits = model(x)
probs = softmax(logits)
loss = cross_entropy(probs, target)
```
- Logits keep numerical range unconstrained; softmax provides normalized probabilities.
- Separation aids clarity and flexibility.

## Mini‑Batch Training
```python
for x_batch, y_batch in DataLoader(dataset, batch_size=32, shuffle=True):
    batch_loss = Value(0.0)
    for x_vals, y in zip(x_batch, y_batch):
        xv = [Value(v) for v in x_vals]
        logits = model(xv)
        probs = softmax(logits if isinstance(logits, list) else [logits])
        batch_loss = batch_loss + cross_entropy(probs, y)
    batch_loss = batch_loss / len(x_batch)
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
```
- Aggregate batch loss (sum/mean) and call backward once per batch.
- Gradients accumulate correctly across samples sharing parameters.

## Regularization
- Dropout:
  - train(): randomly zero activations with probability p; scale kept activations by 1/(1−p)
  - eval(): identity
- Weight Decay:
  - Implemented in optimizers as L2 penalty: grad <- grad + λ·w (weights only)
  - Filters by parameter names to avoid regularizing biases

## Checkpoints
```python
save_checkpoint(model, optimizer, "ckpt.json")
load_checkpoint(model, optimizer, "ckpt.json")
```
- Explicit state dicts for reproducibility and portability (no pickled objects).

## Gradient Checking
- numerical_grad uses centered finite differences per parameter.
- Compares numerical vs analytical grads; reports max difference; fails if above tolerance.

