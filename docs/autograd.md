# Autograd (Reverse‑Mode) Deep Dive

This engine implements scalar reverse‑mode automatic differentiation via a Value class. Each operation produces a new Value node that stores its parents and a local backward closure.

## Core Abstraction: Value
- data: float
- grad: float (accumulated derivative of output w.r.t. this node)
- _prev: set[Value] (parents in the graph)
- _op: str (operation label)
- _backward: callable (computes local gradient contribution)

Supported operations: +, -, *, /, ** (scalar), unary negation, relu, log, exp.

## Building the Graph
- Operator overloading creates new nodes and records parents.
- Each node registers a closure that encodes its local gradient rule.

Example:
```python
z = (x * y + b).relu()
```
- mul node’s backward applies product rule
- add node passes gradient to both inputs
- relu gates gradients by output > 0

## Backward Pass
- backward() performs a DFS to produce a topological ordering of nodes reachable from the output.
- Seeds output.grad = 1.0.
- Traverses nodes in reverse topological order, calling each node’s _backward() exactly once.
- Gradients accumulate (+=) to support fan‑out in the graph.

## Why Reverse‑Mode
- Efficient when number of outputs is small and parameters are many (common in NN training).
- Computes gradients for all parameters in a single backward pass.

## Numerical Safety
- log uses epsilon clamp to avoid log(0).
- exp integrates into autodiff with d/dx exp(x) = exp(x).
- relu blocks gradients when output ≤ 0.

## Gradient Checking
- numerical_grad uses centered finite differences:
  - For each parameter p: (f(p+ε) − f(p−ε)) / (2ε)
  - Compares against analytical gradients from backward()
  - Reports max absolute difference; flags if > tolerance

## Internal Life Cycle
```text
Forward:
  create nodes -> record parents -> set local _backward closures

Backward:
  topo sort -> seed grad at output -> reverse traverse -> run closures
```

