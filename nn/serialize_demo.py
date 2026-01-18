import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.mlp import MLP
from nn.optim import Adam
from nn.functional import softmax, predict
from nn.loss import cross_entropy
from nn.serialization import save_checkpoint, load_checkpoint


def train(model, optimizer, data, epochs=50):
    for _ in range(epochs):
        for x_vals, y in data:
            x = [Value(v) for v in x_vals]
            logits = model(x)
            probs = softmax(logits if isinstance(logits, list) else [logits])
            loss = cross_entropy(probs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    data = [
        ([1.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 1.0], 2),
        ([0.5, 0.5], 2),
    ]

    model = MLP(2, [16, 16, 3])
    optimizer = Adam(model.parameters(), lr=0.03)
    train(model, optimizer, data, epochs=50)

    ckpt_path = os.path.join(os.path.dirname(__file__), "ckpt.json")
    save_checkpoint(model, optimizer, ckpt_path)

    model2 = MLP(2, [16, 16, 3])
    optimizer2 = Adam(model2.parameters(), lr=0.03)
    load_checkpoint(model2, optimizer2, ckpt_path)

    train(model2, optimizer2, data, epochs=50)

    # run preds
    correct = 0
    for x_vals, y in data:
        x = [Value(v) for v in x_vals]
        logits = model2(x)
        probs = softmax(logits if isinstance(logits, list) else [logits])
        if predict(probs) == y:
            correct += 1
    print("Resumed training accuracy:", correct / len(data))


if __name__ == "__main__":
    main()
