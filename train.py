import argparse
import time
import os

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import MetalDataset
from transform import get_transfrom, test_transfrom
from model import MetalModel
from loss import WeightFocalLoss
from utils import cluster2target

def build_argparse():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--exp', help='The index of this experiment', type=int, default=1)
    parser.add_argument('--model_name', default='se_resnext101_32x4d')
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default='Adam')
    
    # FC and Albumentation
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    # Additional Hyperparameter
    parser.add_argument('--lr', type=float,default=0.0001)
    parser.add_argument('--lr_name', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--output_class', type=int, default=15)

    # Loop control
    parser.add_argument('--epoch', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default=16)

    # Additional    
    parser.add_argument('--load_model_para', help='Enter the model.pth file name', type=str, default=None)
    parser.add_argument('--cluster_img', type=bool, default=True)

    return parser

def check_argparse(args):
    assert args.model_name in [
                                'resnet18', 'resnet152', 
                                'densenet121', 'densenet161', 
                                'se_resnet50', 'se_resnet152',
                                'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                'efficientnet-b0', 
                                'efficientnet-b7' 
                                ]

def build_train_val_test_dataset(args):
    train_dataset = MetalDataset(mode='train', cluster_img=args.cluster_img, transform=True, 
                                image_size=args.image_size, val_split=args.val_split, 
                                test_spilt=args.test_split, seed=args.seed)
    val_dataset   = MetalDataset(mode='val', cluster_img=False, image_size=args.image_size, 
                                val_split=args.val_split,test_spilt=args.test_split, seed=args.seed)
    test_dataset  = MetalDataset(mode='test', cluster_img=False, image_size=args.image_size, 
                                val_split=args.val_split, test_spilt=args.test_split, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=os.cpu_count(),batch_size=args.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, pin_memory=True, num_workers=2*os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    test_dataloader   = DataLoader(test_dataset, pin_memory=True, num_workers=2*os.cpu_count(), batch_size=args.batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader

def freeze_pretrain(model, freeze=True):
    if freeze:
        for name, par in model.named_parameters():
            if name.startswith('cnn_model'):
                par.requires_grad = False
    else:
        for name, par in model.named_parameters():
            if name.startswith('cnn_model'):
                par.requires_grad = True

def build_scheduler(optimizer, name, freeze):
    if name == 'ReduceLROnPlateau':
        if freeze == True:
            scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=6)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=2)
        
    elif name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

    return scheduler
  
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    # data
    print('\n-------- Data Preparing --------\n')
    train_dataloader, val_dataloader, test_dataloader = build_train_val_test_dataset(args)
    print('\n-------- Data Preparing Done! --------\n')

    # model
    print('\n-------- Preparing Model --------\n')
    model = MetalModel(model_name = args.model_name, hidden_dim=args.hidden_dim, 
                        activation=args.activation, output_class=args.output_class)
    # freeze CNN pretrained model
    if args.freeze:
        freeze_pretrain(model, True)
    else:
        freeze_pretrain(model, False)

    # loading previous trained model parameters
    # usually for freeze-unfreeze method
    if args.load_model_para:
        model.load_state_dict(torch.load('/home/rico-li/Job/Metal/model_save/'+args.load_model_para))
    else:
        pass

    # pass to CUDA device
    model = model.to(device)
    
    # TODO: need to discuss
    # WeightFocalLoss() for imbalanced dataset (while considering the minor classes are important)
    # rangeloss, for inter-class and intra-class seperation
    criterion = nn.CrossEntropyLoss()

    if args.optim == 'Adam':
        # before acc 80 %
        optimizer = optim.Adam(model.parameters())
    elif args.optim == 'SGD':
        # after acc 80 %
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, nesterov=True, weight_decay=0.01)

    scheduler = build_scheduler(optimizer, args.lr_name, args.freeze)

    print('\n-------- Preparing Model Done! --------\n')

    # train
    print('\n-------- Starting Training --------\n')
    # tensorboard
    writer = SummaryWriter(f'runs/trial_{args.exp}')

    for epoch in range(args.epoch):
        start_time = time.time()

        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')

        #  --------------------------- TRAINING LOOP ---------------------------
        print('---Training Loop begins---')
        optimizer.zero_grad() # using gradient accumulated method to solve batch_size too small issue
        model.train()
        for i, data in enumerate(train_dataloader, start=0):
            input, target = data[0].to(device), data[1].to(device)
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            train_running_loss += loss.item()
            if (i+1)%args.batch_size == 0:  # using gradient accumulated method to solve batch_size too small issue
                                            # real batch size is 16 * 16 = 256
                writer.add_scalar('Batch-Averaged loss', train_running_loss/(args.batch_size), args.batch_size*epoch + int((i+1)/(args.batch_size)))
                print( f"[{epoch+1}, {int((i+1)/(args.batch_size))}]: %.3f" % (train_running_loss/(args.batch_size)) )
                optimizer.step()
                optimizer.zero_grad()
                train_running_loss = 0.0
        
        lr = [group['lr'] for group in optimizer.param_groups]
        print('Epoch:', f'{epoch+1}/{args.epoch}',' LR:', lr[0])
        writer.add_scalar('Learning Rate', lr[0], epoch)

        print('---Training Loop ends---')
        print(f'---Training spend time: %.1f sec' % (time.time() - start_time))
        
        #  --------------------------- VALIDATION LOOP ---------------------------
        with torch.no_grad():
            model.eval()
            val_run_loss = 0.0
            print('\n---Validaion Loop begins---')
            start_time = time.time()
            batch_count = 0
            total_count = 0
            correct_count = 0
            for i, data in enumerate(val_dataloader, start=0):
                input, target = data[0].to(device), data[1].to(device)
                # TODO: only for combine 13 and 11 case
                target[target == 13] = 11
                target[target == 14] = 13

                output = model(input)
                _, predicted = torch.max(output, 1)

                if args.cluster_img:
                    # cluster label to target label
                    output_cluster = cluster2target(output)
                    loss = criterion(output_cluster, target)
                    val_run_loss += loss.item()
                    
                    # cluster label to target label
                    predicted = cluster2target(predicted)
                    correct_count += (predicted == target).sum().item()
                else:
                    loss = criterion(output, target)
                    val_run_loss += loss.item()
                    correct_count += (predicted == target).sum().item()

                batch_count += 1
                total_count += target.size(0)
            accuracy = (100 * correct_count/total_count)
            val_run_loss = val_run_loss/batch_count
            
            if args.lr_name == 'ReduceLROnPlateau':
                scheduler.step(val_run_loss)
            elif args.lr_name == 'StepLR':
                scheduler.step()

            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f"\nLoss of {epoch+1} epoch is %.3f" % (val_run_loss))
            print(f"Accuracy is %.2f %% \n" % (accuracy))
                
            print('---Validaion Loop ends---')
            print(f'---Validaion spend time: %.1f sec' % (time.time() - start_time))
    writer.close()
    print('\n-------- End Training --------\n')
    
    print('\n-------- Saving Model --------\n')
    savepath = f'/home/rico-li/Job/Metal/model_save/{str(args.exp)}_{str(args.model_name)}.pth'
    torch.save(model.state_dict(), savepath)
    print('-------- Saved --------')
    print(f'\n== Trial {args.exp} finished ==\n')


if __name__ == '__main__':

    start_time = time.time()
    main()
    print('--- Execution time ---')
    print(f'--- %.1f sec ---' % (time.time() - start_time))

# --- code snippet ---
# tensorboard --logdir runs/trial_X/
# time python yourprogram.py

# Freeze
# python train.py --exp X --epoch 10 --freeze True --output_class 36
# Unfreeze and load .pth
# python train.py --exp X --epoch 15  --load_model_para 65_se_resnext101_32x4d.pth --output_class 36