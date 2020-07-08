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
                image_size = 256, seed = 42, cluster_img=False, combine=False):
        assert mode in ['train', 'test', 'val']
        super().__init__()

        assert os.getcwd() in ['/home/rico-li/Job/Metal', '/home/aiuser/Job/MetalClassification'], 'in the wrong working directory'
        if os.getcwd() == '/home/rico-li/Job/Metal':
            print('In local machine.')
        else:
            print('In server.')

        path = os.getcwd()+'/Image'
        class_names = os.listdir(path)
        class_names.sort()
        
        label = []
        image_path = []
        img_names = []
        image_path_merge = []
        np.random.seed(seed)  # fix the train val test set.
        for i, class_name in enumerate(class_names):
            image_names = os.listdir(f'{path}/{class_name}')
            img_names += image_names 
            image_path += [f'{path}/{class_name}/{image_name}' for image_name in image_names]

            if cluster_img == False:
                if combine == True:
                    if i == 13:
                        label += [11] * len(image_names) # TODO: class 13 are re-label to class 11
                    elif i == 14:
                        label += [13] * len(image_names) # TODO: class 14 are re-label to class 13
                    else:
                        label += [i] * len(image_names)
                else:
                    label += [i] * len(image_names)

                self.index_list = train_val_test_index(label, mode, val_split, test_spilt)
                self.label = [label[i] for i in self.index_list]
                self.data_path = [image_path[i] for i in self.index_list]
                self.image_names = [img_names[i] for i in self.index_list]

            else:
                # cluster label: for training data only
                assert mode == 'train', 'for training data only'
                # ----- notice here -----
                # the file is created with train_val_test_index distribution, no need to use
                # [label[i] for i in self.index_list] anymore!
                # their images name is in oneDfea_imgname__1113merge.txt
                # ----- notice here -----
                label = torch.load('oneDfea_train_label36')
                label = label.type(torch.LongTensor).tolist()
                self.label = label
                imgnamefile = open('oneDfea_imgname_1113merge.txt')
                imgName = imgnamefile.read()
                imgName = imgName.splitlines() # list
                self.image_names = imgName

                name_length = len(class_name)
                if class_name == 'TTA':
                    image_path_merge += [f'{path}/{class_name}/{image_name}' for image_name in imgName if class_name == image_name[:name_length]]
                    image_path_merge += [f'{path}/TTP/{image_name}' for image_name in imgName if 'TTP' == image_name[:name_length]]
                elif class_name == 'TTP':
                    pass
                else:
                    image_path_merge += [f'{path}/{class_name}/{image_name}' for image_name in imgName if class_name == image_name[:name_length]]
                
                self.data_path = image_path_merge
        
        self.mode = mode
        self.transform = transform
        self.image_size = image_size
            
    def __len__(self):
        return len(self.data_path)

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

class EncoderDataset(Dataset):
    def __init__(self, mode, val_en_spilt = 0.2, val_an_spilt = 0.25, image_size = 256, seed = 42):
        super().__init__()

        assert mode in ['training', 'val_en', 'threshold','val_an']
        self.image_size = image_size
        self.mode = mode
        self.seed = seed
        path = os.getcwd()+'/Image'
        class_names = os.listdir(path)
        class_names.sort()

        label = []
        image_path = []
        img_names = []

        # for label 10
        label_10 = []
        image_path_10 = []
        img_names_10 = []

        for i, class_name in enumerate(class_names):
            if class_name != 'TPI':
                image_names = os.listdir(f'{path}/{class_name}')
                img_names += image_names 
                image_path += [f'{path}/{class_name}/{image_name}' for image_name in image_names]
                label += [i] * len(image_names)
            else:
                image_names_10 = os.listdir(f'{path}/{class_name}')
                img_names_10 += image_names_10 
                image_path_10 += [f'{path}/{class_name}/{image_name}' for image_name in image_names_10]
                label_10 += [i] * len(image_names_10)
        
        if mode in ['training','val_en']:
            self.index_list = self.setSpilt(label, val_en_spilt, mode)
            self.data_path = [image_path[i] for i in self.index_list]
            self.label = [label[i] for i in self.index_list]
            self.image_names = [img_names[i] for i in self.index_list]
        
        elif mode in ['threshold','val_an']:
            self.index_list_10 = self.setSpilt(label_10, val_an_spilt, mode)
            self.data_path_10 = [image_path_10[i] for i in self.index_list_10]
            self.label_10 = [label_10[i] for i in self.index_list_10]
            self.image_names_10 = [img_names_10[i] for i in self.index_list_10]

    
    def __len__(self):
        if self.mode in ['training','val_en']:
            return len(self.image_names)
        elif self.mode in ['threshold','val_an']:
            return len(self.image_names_10)

    def __getitem__(self, idx):
        if self.mode in ['training','val_en']:
            label = torch.tensor(self.label[idx])
            path_i = self.data_path[idx]
            image = cv2.imread(path_i, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_name = self.image_names[idx]
            image = test_transfrom(image, size=self.image_size)

            return image, label, image_name

        elif self.mode in ['threshold','val_an']:
            label = torch.tensor(self.label_10[idx])
            path_i = self.data_path_10[idx]
            image = cv2.imread(path_i, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_name = self.image_names_10[idx]
            image = test_transfrom(image, size=self.image_size)

            return image, label, image_name

    def setSpilt(self, label, spilt, whichindex):
        label = np.array(label)
        index_list_uni = np.unique(label)
        train_index = []
        val_index = []

        np.random.seed(self.seed)
        for i in index_list_uni:
            index = np.where(label == i)
            class_index = np.random.permutation(index[0]).tolist()
            train_index += class_index[:round(len(class_index)*(1-spilt))]
            val_index += class_index[round(len(class_index)*(1-spilt)): ]

        if whichindex in ['training', 'threshold']:
            return train_index
        elif whichindex in ['val_en','val_an']:
            return val_index



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    train_dataset = MetalDataset(mode='test', transform=True, cluster_img=False, image_size=256, val_split=0.3, test_spilt=0.1, seed=42)
    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=os.cpu_count(), batch_size=16, shuffle=True)
    labels = torch.Tensor().type(torch.long)
    for i, (image, label, image_name) in enumerate(train_dataloader):
        labels = torch.cat((labels, label))
        print(label)
        print(f"{i}-batch")

    labels = labels.numpy()
    plt.hist(labels, bins=100, alpha=0.75)
    plt.show()



    # from matplotlib import pyplot as plt
    # from torchvision.utils import make_grid
    # from torch.utils.tensorboard import SummaryWriter

    # writer = SummaryWriter(f'runs/test_10')

    # dataset = EncoderDataset(mode='training')
    # dataloader = DataLoader(dataset, batch_size=6, shuffle=False)
    # start_time = time.time()
    # for image, label, image_name in dataloader:
    #     image = image[:6, ...] # (B, C, H, W)

    #     mean =[0.3835, 0.3737, 0.3698]
    #     std= [1.0265, 1.0440, 1.0499]

    #     for i in range(3):
    #         image[:,i,...] = 255 - ((image[:,i,...] * std[i] + mean[i])*255)
                
    #     image = image.type(torch.int8)
    #     image = make_grid(image, nrow=3) # (C, H, W)
    #     img_np = image.numpy().transpose(1,2,0) # (H, W, C)
    #     plt.imshow(img_np)
    #     plt.show()

    #     writer.add_image("Image", image)
    #     break
    #     # origi_img = input[:n,...].clone().detach() #(n, C, H, W)
    #     # decor_img = model(origi_img) #(n, C, H, W)
    #     # img = torch.cat((origi_img, decor_img), dim=0) #(n, C, H, W)
    #     # img = make_grid(img, nrow=n)
    #     # writer.add_image(f"Original-Up, decor-Down in epoch: {epoch+1}", img, dataformats='CHW')
    # writer.close()