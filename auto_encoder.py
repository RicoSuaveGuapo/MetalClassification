# from pytorch-beginner/08-AutoEncoder/conv_autoencoder.py by L1aoXingyu
import math
import os
import argparse
import time
import cv2
import numpy as np

import torch
import torchvision
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import make_grid

from dataset import MetalDataset
from transform import get_transfrom, test_transfrom
from train import build_argparse, check_argparse
from dataset import EncoderDataset
from torch.utils.tensorboard import SummaryWriter

from vae import VAE, loss_function


def outSize(img, kernal, stride, padding, transpose = False):
    if transpose == False:
        outsize = math.floor((img + 2 * padding - kernal)/stride +1)
    else:
        outsize = (img - 1) * stride - 2 * padding + kernal 

    return outsize


# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=2, padding=1),  # b, 32, 128
#             nn.ReLU(True),
#             nn.MaxPool2d(2),  # b, 32, 64
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),  # b, 64, 32
#             nn.ReLU(True),
#             nn.MaxPool2d(2),  # b, 64, 16
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),  # b, 128, 8
#             nn.ReLU(True),
#             nn.MaxPool2d(2),  # b, 128, 4, 4
#             nn.Conv2d(128, 128, 3, stride=1),  # b, 128, 2, 2
#             nn.ReLU(True),
#         )

#         self.fc_en = nn.Linear(128*2*2, 16)
#         self.act_1 = nn.ReLU(True)
#         self.fc_de = nn.Linear(16, 128*2*2)
#         self.act_2 = nn.ReLU(True)

#         self.decoder = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 128, 4
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(128, 128, 3, stride=1, padding=0),  # b, 128, 4
#             nn.ReLU(True),

#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 128, 8
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(128, 64, 3, stride=1, padding=0),  # b, 64, 8
#             nn.ReLU(True),

#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 64, 16
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(64, 32, 3, stride=1, padding = 0),  # b, 32, 16
#             nn.ReLU(True),

#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 32, 32
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(32, 16, 3, stride=1, padding = 0),  # b, 16, 32
#             nn.ReLU(True),

#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 16, 64
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(16, 8, 3, stride=1, padding = 0),  # b, 8, 64
#             nn.ReLU(True),

#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 8, 128
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(8, 4, 3, stride=1, padding = 0),  # b, 4, 128
#             nn.ReLU(True),
            
#             nn.Upsample(scale_factor=2, mode='bilinear'), # b, 4, 256
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(4, 3, 3, stride=1, padding = 0),  # b, 3, 256
#             nn.Tanh()
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = x.view(-1, 128*2*2)
#         x = self.act_1(self.fc_en(x))
#         x = self.act_2(self.fc_de(x))
#         x = x.view(-1,128,2,2)
#         x = self.decoder(x)
#         return x


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=3, padding=1),  # b, 16, 86
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 43 
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # b, 8, 22
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 21
        )
        self.fc_en = nn.Linear(32*21*21, 10)
        self.act = nn.ReLU(True)
        self.dropout = nn.Dropout()
        self.fc_de = nn.Linear(10, 32*21*21)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2),  # b, 16, 5, 5 #43
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15 #129
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 2, stride=2, padding=1),  # b, 1, 28, 28 #256
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc_en(x))
        x = self.dropout(x)
        x = self.act(self.fc_de(x))
        x = x.view(-1, 8, 21, 21)
        x = self.decoder(x)
        return x
    

def build_scheduler(optimizer, name, freeze):
    if name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=2)
        
    elif name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    return scheduler

# def build_optim(parameters, loss):
#     if args.optim == 'Adam':
#         optimizer = optim.Adam(model.parameters())
#     elif args.optim == 'SGD':
#         optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, nesterov=True, weight_decay=0.01)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    print('\n-------- Data Preparing --------\n')
    traindataset = EncoderDataset(mode='training')
    # val_en_dataset = EncoderDataset(mode='val_en')
    # threshold_dataset = EncoderDataset(mode='threshold')
    # val_an_dataset = EncoderDataset(mode='val_an')

    trainloader = DataLoader(traindataset, num_workers=os.cpu_count(), pin_memory=True, batch_size=args.batch_size, shuffle=True)
    # val_en_loader = DataLoader(val_en_dataset, num_workers=os.cpu_count(), pin_memory=True, batch_size=args.batch_size, shuffle=True)
    # threshold_loader = DataLoader(threshold_dataset, num_workers=os.cpu_count(), pin_memory=True, batch_size=args.batch_size, shuffle=True)
    # val_an_loader = DataLoader(val_an_dataset, num_workers=os.cpu_count(), pin_memory=True, batch_size=args.batch_size, shuffle=True)
    print('\n-------- Data Preparing Done! --------\n')
    
    print('\n-------- Preparing Model --------\n')
    model = autoencoder()
    # model = VAE()
    model = model.to(device)
    print('\n-------- Preparing Model Done! --------\n')
    
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters())
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, nesterov=True, weight_decay=0.01)

    # TODO: maybe add in weight
    criterion = nn.MSELoss()
    # reconstruction_function = nn.MSELoss(size_average=False)

    scheduler = build_scheduler(optimizer, args.lr_name, args.freeze)

    writer = SummaryWriter(f'runs/auto_encoder_trial_{args.exp}')

    for epoch in range(args.epoch):
        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')

        batch_count = 0
        for i, data in enumerate(trainloader, start=0):

            optimizer.zero_grad()
            input = data[0].to(device)
            # ===================forward=====================
            # input = input.view(input.size(0),-1)
            # recon_batch, mu, logvar = model(input)
            # loss = loss_function(recon_batch, input, mu, logvar)
            output = model(input)
            loss = criterion(output, input)
            # ===================backward====================
            loss.backward()
            train_running_loss += loss.item()
            optimizer.step()
            # ===================batch log========================
            print( f"[{epoch+1}, {int(i+1)}]: %.3f" % (loss.item()) )
            writer.add_scalar('Batch-Averaloss', loss.item(), batch_count*epoch + i)
            
        batch_count = i 
        # ===================epoch log========================
        lr = [group['lr'] for group in optimizer.param_groups]
        print('Epoch:', epoch+1,'LR:', lr[0])
        writer.add_scalar('Learning Rate', lr[0], epoch)

        print('epoch [{}/{}], averaged loss:{:.3f}'
            .format(epoch+1, args.epoch, train_running_loss/batch_count))
        
        if args.lr_name == 'ReduceLROnPlateau':
            scheduler.step(train_running_loss/batch_count)
        elif args.lr_name == 'StepLR':
            scheduler.step()

        with torch.no_grad():
            # number of images to show
            n = 6
            origi_img = input[:n,...].clone().detach() #(n, C, H, W)
            decor_img = model(origi_img) #(n, C, H, W)
            img = torch.cat((origi_img, decor_img), dim=0) #(n, C, H, W)
            mean =[0.3835, 0.3737, 0.3698]
            std= [1.0265, 1.0440, 1.0499]

            for i in range(3):
                img[:,i,...] = 255 - ((img[:,i,...] * std[i] + mean[i])*255)
                
            img = img.type(torch.int8)
            img = make_grid(img, nrow=n)

            writer.add_image(f"Original-Up, decor-Down in epoch: {epoch+1}", img, dataformats='CHW')

    writer.close()

    print('\n-------- Saving Model --------\n')
    savepath = f'/home/rico-li/Job/Metal/model_save/{str(args.exp)}_autoencoder.pth'
    torch.save(model.state_dict(), savepath)
    print('-------- Saved --------')


    print(f'\n======auto_encoder_trial_{args.exp}======\n')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f'\n--- %.1f sec ---\n' % (time.time() - start_time))