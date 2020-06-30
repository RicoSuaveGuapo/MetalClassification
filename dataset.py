import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from transform import get_transfrom, test_transfrom
from utils import train_val_test_index

class MetalDataset(Dataset):

    def __init__(self, mode='train', transform = None, val_split = 0.3, test_spilt = 0.1,
                image_size = 256, seed = 42, cluster_img=False):
        assert mode in ['train', 'test', 'val']
        super().__init__()

        assert os.getcwd() == '/home/rico-li/Job/Metal', 'in the wrong working directory'
        path = os.getcwd()+'/Image' if not cluster_img else os.getcwd()+'/stealimage_clustered'
        class_names = os.listdir(path)
        class_names.sort()
        
        label = []
        image_path = []
        img_names = [] # save the image names
        for i, class_name in enumerate(class_names):
            image_names = os.listdir(f'{path}/{class_name}')
            img_names += image_names 
            image_path += [f'{path}/{class_name}/{image_name}' for image_name in image_names]
            label += [i] * len(image_names) # change to new cluster labels

        self.mode = mode
        self.transform = transform
        self.image_size = image_size

        np.random.seed(seed)
        self.index_list = train_val_test_index(label, mode, val_split, test_spilt)
        self.data_path = [image_path[i] for i in self.index_list]

        # cluster label
        if mode == 'train':
            label = torch.load('oneDfea_train_label37')
            label = label.type(torch.LongTensor).tolist()
            self.label = label
        else:
            self.label = [label[i] for i in self.index_list]

        self.image_names = [img_names[i] for i in self.index_list]
        
            
    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        label = torch.tensor(self.label[idx])
        path_i = self.data_path[idx]
        image = cv2.imread(path_i, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_name = self.image_names[idx]
        
        if self.transform:
            image = get_transfrom(image, crop_size=self.image_size) # tensor
        else:
            image = test_transfrom(image, size=self.image_size) # tensor

        return image, label, image_name
        

if __name__ == '__main__':
    dataset = MetalDataset(mode='val', transform=True, cluster_img=False)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
    start_time = time.time()
    for image, label, image_name in dataloader:
        print(f'---One batch spends time: %.1f sec' % (time.time() - start_time))
        print(image.shape)
        print(label)
        break

    # a = [0]*5 + [1]*4
    # a = np.array(a)
    # a_uni = np.unique(a)
    # train_index = []
    # val_index = []
    # test_index = []    