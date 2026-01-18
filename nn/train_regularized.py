import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from autograd.value import Value
from nn.mlp import MLP
from nn.optim import Adam
from nn.functional import softmax, predict, accuracy
from nn.loss import cross_entropy
from nn.dataloader import Dataset, DataLoader


def main():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    Y = [0, 1, 2, 2]
    dataset = Dataset(X, Y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=123)

    model = MLP(2, [64, 64, 3], dropout=0.5, dropout_seed=7)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer.attach_param_names([name for name, _ in model.named_parameters()])

    model.train()
    for epoch in range(300):
        total_loss = Value(0.0)
        correct = 0
        for x_batch, y_batch in loader:
            batch_loss = Value(0.0)
            for x_vals, y in zip(x_batch, y_batch):
                x = [Value(v) for v in x_vals]
                logits = model(x)
                probs = softmax(logits if isinstance(logits, list) else [logits])
                batch_loss = batch_loss + cross_entropy(probs, y)
                if predict(probs) == y:
                    correct += 1
            batch_loss = batch_loss / len(x_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss = total_loss + batch_loss
        if epoch % 50 == 0:
            print(f"train epoch {epoch}, loss {total_loss.data:.3f}, acc {correct / len(dataset):.2f}")

    model.eval()
    acc = accuracy(model, list(zip(X, Y)))
    print("eval acc", acc)


if __name__ == "__main__":
    main()
