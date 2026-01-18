from typing import List, Tuple, Iterator, Union
import random
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Dataset:
    def __init__(self, X: List[List[float]], Y: List[Union[float, int]]):
        assert len(X) == len(Y)
        self.X = X
        self.Y = Y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[List[float], Union[float, int]]:
        return self.X[idx], self.Y[idx]


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = True, seed: Union[int, None] = None):
        assert batch_size >= 1
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __iter__(self) -> Iterator[Tuple[List[List[float]], List[Union[float, int]]]]:
        idxs = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for start in range(0, len(idxs), self.batch_size):
            batch_idx = idxs[start:start + self.batch_size]
            x_batch: List[List[float]] = []
            y_batch: List[Union[float, int]] = []
            for i in batch_idx:
                x, y = self.dataset[i]
                x_batch.append(x)
                y_batch.append(y)
            yield x_batch, y_batch
