import time
import os 

import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder


def train_val_test_index(label: list, whichindex: str, val_split: float, test_split: float)-> list:
    label = np.array(label)
    index_list_uni = np.unique(label)
    train_index = []
    val_index = []
    test_index = []

    for i in index_list_uni:
        index = np.where(label == i)
        class_index = np.random.permutation(index[0]).tolist()
        train_index += class_index[:round(len(class_index)*(1-val_split-test_split))]
        val_index += class_index[round(len(class_index)*(1-val_split-test_split)): round(len(class_index)*(1-test_split))]
        test_index += class_index[round(len(class_index)*(1-test_split)):]
    
    if whichindex == 'train':
        return train_index
    elif whichindex == 'val':
        return val_index
    else:
        return test_index


def imageFoldercv2():
    original_path = os.getcwd()
    assert original_path == '/home/rico-li/Job/Metal', 'note that working directory'

    image_path = original_path+'/Image'
    class_names = os.listdir(image_path)
    class_names.sort()
    
    label_i = 0
    data = []
    for class_name in class_names:
        class_time = time.time()

        image_names = os.listdir(image_path+f'/{class_name}')

        images = [cv2.imread(f'{image_path}/{class_name}/{image_name}', cv2.IMREAD_COLOR) for image_name in image_names]
        labels = [label_i] * len(images)

        data += [ [i,j] for i, j in zip(images, labels)]
        label_i += 1
        print(f'-- class time: %.2f' % (time.time() - class_time))

    return data
    # desired output shape 
    # [[image_1: numpy array , label_1: number], [image_2, label_2], ...] 



class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x * (torch.tanh(F.softplus(x)))


# this resize might discard
# def resize(img, size=128):
#     # resize and transpose some image with wrong xy
#     # input img: [CHW]
#     assert img.shape[0] < 4, 'input img shape should be (CHW)'

#     # rotate some images with different direction
#     if img.shape[1] < img.shape[2]:
#         img = img.transpose(0, 2, 1)
#     else:
#         pass

#     assert isinstance(size, int), 'size should be int'
#     size = (size, size)
#     # cv2 format is (HWC)
#     img = img.transpose(1,2,0)
#     img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
#     # change back to (CHW)
#     img = img.transpose(2,0,1).copy() # make the memory contiguous
#     return img



def index2label(index=None, inverse=False):
    # change index (0~14) to label (GB, PS, etc)
    # inverse: label2index
    
    index_dict = {'GB': 0, 'PS': 1, 'SDA': 2, 'SDB': 3, 'SDC': 4, 
            'T1H': 5, 'T2H': 6, 'THF': 7, 'THK': 8, 'THS': 9, 
            'TPI': 10, 'TTA': 11, 'TTC': 12, 'TTP': 13, 'TTR': 14}
    
    if not inverse:
        index_dict = {v: k for k, v in index_dict.items()}
        label = index_dict[index]
    else:
        label = index_dict[index]

    return label

# def read_df(path):
#     df = pd.read_csv(path)['label']
#     df = pd.get_dummies(df)
#     df = df.to_numpy()
#     df = torch.from_numpy(df)
#     return df


if __name__ == '__main__':
    # img = np.random.randn(3, 256, 256)
    # img_resize = resize(img)
    # print(img_resize.shape)

    # df = read_df('metalData.csv')
    # print(type(df[0]))
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
        print(image_path[-1])

        label += [label_i] * len(image_names)
        label_i += 1
    