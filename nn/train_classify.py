import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.mlp import MLP
from nn.optim import Adam
from nn.functional import softmax, predict, accuracy
from nn.loss import cross_entropy


def main():
    data = [
        ([1.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 1.0], 2),
        ([0.5, 0.5], 2),
    ]

    model = MLP(2, [16, 16, 3])
    optimizer = Adam(model.parameters(), lr=0.03)

    for epoch in range(300):
        correct = 0
        total_loss = Value(0.0)

        for x_vals, y in data:
            x = [Value(v) for v in x_vals]
            logits = model(x)
            probs = softmax(logits if isinstance(logits, list) else [logits])
            loss = cross_entropy(probs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss
            if predict(probs) == y:
                correct += 1

        if epoch % 50 == 0:
            print(f"epoch {epoch}, loss {total_loss.data:.3f}, acc {correct / len(data):.2f}")

    print("Final accuracy:", accuracy(model, data))


if __name__ == "__main__":
    main()
