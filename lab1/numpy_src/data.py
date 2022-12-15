import random

import numpy as np



class Dataset:

    def __init__(self, data: np.ndarray, target: np.ndarray):
        self.data = data
        self.target = target

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def __len__(self):
        return self.data.shape[0]


class Dataloader:

    def __init__(self, dataset: Dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.__dataset_len = len(self.dataset)
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.idxs = list(range(int(np.ceil(self.__dataset_len / self.__batch_size))))

    def __getitem__(self, idx):
        index = self.idxs[idx] * self.__batch_size
        to = min(self.__dataset_len - index, self.__batch_size)
        samples = [self.dataset[index + batch_i] for batch_i in range(to)]
        batch = list()
        for i in range(len(samples[0])):
            sample = list()
            for j in range(len(samples)):
                sample.append(samples[j][i])
            batch.append(np.array(sample))

        return batch

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.__batch_size))

    def __iter__(self):
        if self.__shuffle:
            random.shuffle(self.idxs)

        for i in range(self.__len__()):
            yield self.__getitem__(i)
