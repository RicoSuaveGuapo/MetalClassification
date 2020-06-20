import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transform import get_transfrom, test_transfrom

class MetalDataset(Dataset):

    def __init__(self, mode='train', transform = None, val_split = 0.3, test_spilt = 0.1,
                image_size = 256, seed = 42):
        assert mode in ['train', 'test', 'val']
        super().__init__()

        assert os.getcwd() == '/home/rico-li/Job/Metal', 'in the wrong working directory'
        path = os.getcwd()+'/Image'
        class_names = os.listdir(path)
        class_names.sort()
        
        label_i = 0
        label = []
        image_path = []
        for class_name in class_names:
            image_names = os.listdir(f'{path}/{class_name}')
            image_path += [f'{path}/{class_name}/{image_name}' for image_name in image_names]
            label += [label_i] * len(image_names)
            label_i += 1

        self.data_path = image_path
        self.label = label

        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        train = mode in ['train', 'val']

        np.random.seed(seed)
        index_list = list(range(len(self.label)))

        # TODO: should include class distribution into the train/val/test set.
        index_list = np.random.permutation(index_list)

        if train:
            split_1 = int(len(self.label) * (1 - val_split - test_spilt))
            split_2 = int(len(self.label) * (1 - test_spilt))

            if self.mode == 'train':
                self.index_list = index_list[:split_1]
            elif self.mode == 'val':
                self.index_list = index_list[split_1:split_2]
        else:
            split_2 = int(len(self.label) * (1 - test_spilt))
            self.index_list = index_list[split_2:]


    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        path_i = self.data_path[idx]
        image = cv2.imread(path_i, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = get_transfrom(image, crop_size=self.image_size) # tensor
        else:
            image = test_transfrom(image, size=self.image_size) # tensor

        return image, label

if __name__ == '__main__':
    dataset = MetalDataset(mode='val', transform=True)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    for image, label in dataloader:
        print(image.shape)
        print(label)
        break
