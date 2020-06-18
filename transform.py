from albumentations.augmentations.transforms import Flip, Normalize, Blur, Resize, RandomCrop
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import Compose
import cv2
from matplotlib import pyplot as plt
import torch
import numpy as np

# TODO: perspective transformation

def get_transfrom(image, size=512, crop_size=256):
    transform = Compose([ Resize(size,size, interpolation=cv2.INTER_AREA),
                        RandomCrop(crop_size, crop_size),
                        Normalize(mean =[0.3835, 0.3737, 0.3698], std= [1.0265, 1.0440, 1.0499]), 
                        Flip(),
                        ToTensorV2()
    ])
    image_transform = transform(image = image)['image']

    return image_transform

def test_transfrom(image, size=256): # note that test image size is same as crop_size in get_transfrom
    transform = Compose([ Resize(size,size, interpolation=cv2.INTER_AREA),
                        Normalize(mean =[0.3835, 0.3737, 0.3698], std= [1.0265, 1.0440, 1.0499]),
                        ToTensorV2()
    ])
    image_transform = transform(image = image)['image']

    return image_transform





if __name__ == '__main__':
    img = cv2.imread("Image/GB/GB_Q0685-1090323111531669.jpg", cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_trans = get_transfrom(img)
    print(img_trans.shape)
    img_trans = img_trans.numpy().transpose(1,2,0)
    
    plt.imshow(img_trans)
    plt.show()
    
    # img = cv2.imread("Image/GB/GB_Q2591-1090325164615389.jpg", cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # img_trans = test_transfrom(img)
    # img_trans = img_trans.numpy().transpose(1,2,0)[...,0].reshape(-1)
    
    # plt.hist(img_trans)
    # plt.show()
