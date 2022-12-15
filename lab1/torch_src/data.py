import os
import random

import cv2
from PIL import Image
from mat4py import loadmat


class CarDataset:
    def __init__(self, root: str, img_folder: str, transform: None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_folder = img_folder
        self.img_path = os.path.join(root, img_folder)
        anno = loadmat(os.path.join(root, f'{img_folder}_annos.mat'))
        print(anno['annotations'].keys())
        self.classes = anno['annotations']['class']
        self.img_names = anno['annotations']['fname']
        self.bbox_x1 = anno['annotations']['bbox_x1']
        self.bbox_x2 = anno['annotations']['bbox_x2']
        self.bbox_y1 = anno['annotations']['bbox_y1']
        self.bbox_y2 = anno['annotations']['bbox_y2']
        self.indexes = list(range(len(self.classes)))
        self.label_names = loadmat(os.path.join('C:\\MySpace\\Projects\\PTDL\\lab1\\data\\LR1-1', 'cars_meta.mat'))[
            'class_names']

    def split_dataset(self, train_part: float):
        train_number = int(train_part * self.__len__())
        train_indexes = random.sample(self.indexes, train_number)
        test_indexes = list(set(self.indexes) - set(train_indexes))

        train_dataset = CarDataset(root=self.root, img_folder=self.img_folder, transform=self.transform)
        train_dataset.indexes = train_indexes

        test_dataset = CarDataset(root=self.root, img_folder=self.img_folder, transform=self.transform)
        test_dataset.indexes = test_indexes

        return train_dataset, test_dataset

    def __getitem__(self, item):
        index = self.indexes[item]
        image = Image.open(os.path.join(self.img_path, self.img_names[index])).convert('RGB')
        # image.crop(box=(self.bbox_x1[index], self.bbox_y1[index], self.bbox_x2[index], self.bbox_y2[index]))

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.classes[index]) - 1
        return image, label

    def __len__(self):
        return len(self.indexes)
