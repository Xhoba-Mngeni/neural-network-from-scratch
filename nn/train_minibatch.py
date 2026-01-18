import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.mlp import MLP
from nn.optim import Adam
from nn.functional import softmax, predict
from nn.loss import cross_entropy, mse_loss
from nn.dataloader import Dataset, DataLoader


def train_classification_minibatch():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    Y = [0, 1, 2, 2]
    dataset = Dataset(X, Y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=42)

    model = MLP(2, [16, 16, 3])
    optimizer = Adam(model.parameters(), lr=0.03)

    for epoch in range(200):
        total_loss = Value(0.0)
        correct = 0
        for x_batch, y_batch in loader:
            preds = []
            batch_loss = Value(0.0)
            for x_vals, y in zip(x_batch, y_batch):
                xv = [Value(v) for v in x_vals]
                logits = model(xv)
                probs = softmax(logits if isinstance(logits, list) else [logits])
                preds.append(probs)
                batch_loss = batch_loss + cross_entropy(probs, y)
            batch_loss = batch_loss / len(x_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss = total_loss + batch_loss
            for probs, y in zip(preds, y_batch):
                if predict(probs) == y:
                    correct += 1
        if epoch % 50 == 0:
            print(f"epoch {epoch}, loss {total_loss.data:.3f}, acc {correct / len(dataset):.2f}")


def train_regression_minibatch():
    X = [[1.0, 2.0, 3.0], [2.0, 0.0, -1.0], [3.0, 1.0, 2.0], [0.0, -1.0, 1.0]]
    Y = [2.0, 1.0, 4.0, 2.0]
    dataset = Dataset(X, Y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=123)

    model = MLP(3, [4, 4, 1])
    optimizer = Adam(model.parameters(), lr=0.03)

    for epoch in range(200):
        total_loss = Value(0.0)
        for x_batch, y_batch in loader:
            batch_loss = Value(0.0)
            for x_vals, y in zip(x_batch, y_batch):
                xv = [Value(v) for v in x_vals]
                y_pred = model(xv)
                if isinstance(y_pred, list):
                    y_pred = y_pred[0]
                batch_loss = batch_loss + mse_loss([y_pred], [y])
            batch_loss = batch_loss / len(x_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss = total_loss + batch_loss
        if epoch % 50 == 0:
            print(f"epoch {epoch}, loss {total_loss.data:.3f}")


if __name__ == "__main__":
    print("Classification minibatch training")
    train_classification_minibatch()
    print("Regression minibatch training")
    train_regression_minibatch()
