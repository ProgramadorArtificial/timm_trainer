import os

import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None, return_labels_onehot=True, load_images_memory=False, onehot=None):
        """
        Args:
            dataset_path (string): Path to the dataset
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None
            return_labels_onehot (bool, optional): If True, returns onehot encoded labels. If False, returns index of
              the labels. Defaults to True
            load_images_memory (bool, optional): If True, load all images in memory. Defaults to False
            onehot (OneHotEncoder, optional): If OneHotEncoder use it, otherwise create a new one. Defaults to None
        """
        self.transform = transform
        self.return_labels_onehot = return_labels_onehot
        self.load_images_memory = load_images_memory

        self.images = []
        self.labels = []
        self.label_to_idx_images = {}
        self.label_to_id = {}
        # Load images and labels (folder name == label)
        count_id = 0
        for cls in os.listdir(dataset_path):
            idxs = []
            for path in [f'{dataset_path}/{cls}/{img}' for img in os.listdir(f'{dataset_path}/{cls}')]:
                if load_images_memory:
                    self.images.append(Image.fromarray(cv2.imread(path)))
                else:
                    self.images.append(path)
                self.labels.append(cls)
                idxs.append(len(self.images)-1)
            if len(idxs) > 0:
                self.label_to_idx_images[self.labels[-1]] = idxs
                self.label_to_id[cls] = count_id
                count_id += 1
        if onehot:
            self.onehot = onehot
        else:
            self.onehot = OneHotEncoder(sparse_output=False)
            self.onehot.fit(np.array([self.labels]).reshape(-1, 1))

    def __len__(self):
        return len(self.labels)

    def get_num_classes(self):
        return sum(len(category) for category in self.onehot.categories_)

    def get_onehot(self):
        return self.onehot

    def get_onehot_classes(self):
        class_names = self.onehot.categories_
        return {index: category for index, category in enumerate(class_names[0])}

    def get_image(self, idx):
        if self.load_images_memory:
            return self.images[idx]
        else:
            return Image.fromarray(cv2.imread(self.images[idx]))

    def __getitem__(self, idx):
        """
        Dataset to load images where folder name is the class and all images inside are the same class
        """
        img = self.get_image(idx)
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        if self.return_labels_onehot:
            label = self.onehot.transform([[label]])
            return img, torch.tensor(label).squeeze()
        else:
            label = self.label_to_id[label]
            return img, torch.tensor(label)
