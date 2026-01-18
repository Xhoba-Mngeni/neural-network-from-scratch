import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.mlp import MLP
from nn.loss import mse_loss
from nn.optim import Adam


def main():
    data = [
        ([1.0, 2.0, 3.0], 2.0),
        ([2.0, 0.0, -1.0], 1.0),
        ([3.0, 1.0, 2.0], 4.0),
        ([0.0, -1.0, 1.0], 2.0),
    ]

    model = MLP(3, [4, 4, 1])
    optimizer = Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        total_loss = Value(0.0)

        for x_vals, y_true in data:
            x = [Value(v) for v in x_vals]
            y_pred = model(x)
            if isinstance(y_pred, list):
                y_pred = y_pred[0]
            loss = mse_loss([y_pred], [y_true])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss

        if epoch % 20 == 0:
            print(epoch, total_loss.data)


if __name__ == "__main__":
    main()
