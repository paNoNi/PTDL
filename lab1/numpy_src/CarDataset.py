import random

import numpy as np
from mat4py import loadmat
import os
from PIL import Image


class CarDataset:
    def __init__(self, root: str, img_folder: str, size: tuple):
        super().__init__()
        self.root = root
        self.size = size
        self.img_folder = img_folder
        self.img_path = os.path.join(root, img_folder)
        anno = loadmat(os.path.join(root, f'{img_folder}_annos.mat'))
        self.classes = anno['annotations']['class']
        self.img_names = anno['annotations']['fname']
        self.indexes = list(range(len(self.classes)))

    def split_dataset(self, train_part: float):
        train_number = int(train_part * self.__len__())
        train_indexes = random.sample(self.indexes, train_number)
        test_indexes = list(set(self.indexes) - set(train_indexes))

        train_dataset = CarDataset(root=self.root, img_folder=self.img_folder, size=self.size)
        train_dataset.indexes = train_indexes
        test_dataset = CarDataset(root=self.root, img_folder=self.img_folder, size=self.size)
        test_dataset.indexes = test_indexes
        return train_dataset, test_dataset

    def __getitem__(self, item):
        index = self.indexes[item]
        image = Image.open(os.path.join(self.img_path, self.img_names[index])).convert('RGB')
        image = image.resize(self.size)
        image = np.asarray(image)
        image = np.transpose(image, axes=(2, 0, 1))
        label = int(self.classes[index]) - 1
        return image, label

    def __len__(self):
        return len(self.indexes)
