import argparse
import time
import os

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from dataset import MetalDataset
from transform import get_transfrom, test_transfrom
from model import MetalModel
from loss import WeightFocalLoss

def build_argparse():
    parser = argparse.ArgumentParser()
    # Basic
    parser.add_argument('--exp', help='The index of this experiment', type=int, default=1)
    parser.add_argument('--model_name', default='resnet18')
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--val_split', type=float, default=0.3)
    parser.add_argument('--test_split', type=float, default=0.1)
    
    # FC and Albumentation
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)

    # Hyperparameter
    parser.add_argument('--learning_rate', type=float,default=0.01)
    parser.add_argument('--freeze', type=bool, default=False)
    # parser.add_argument('--lr_scheduler', default=)

    # Loop control
    parser.add_argument('--epoch', type=int, default = 1)
    parser.add_argument('--batch_size', type=int, default=128)

    # Additional    
    parser.add_argument('--test_section', type=bool, default=False)
    parser.add_argument('--load_model_para', help='Enter the model.pth file name', default=False)

    return parser



def check_argparse(args):
    assert args.model_name in [
                                'resnet18', 'resnet152', # out_channels = 64
                                'densenet121', 'densenet161', # out_channels = 64
                                'se_resnet50', 'se_resnet152',
                                'se_resnext50_32x4d', 'se_resnext101_32x4d',
                                'efficientnet-b0', # out_channels = 32
                                'efficientnet-b7' # out_channels = 64
                                ]



def build_train_val_test_dataset(args):
    train_dataset = MetalDataset(mode='train', image_size=args.image_size, val_split=args.val_split, test_spilt=args.test_split, seed=args.seed)
    val_dataset   = MetalDataset(mode='val', image_size=args.image_size, val_split=args.val_split, test_spilt=args.test_split, seed=args.seed)
    test_dataset  = MetalDataset(mode='test', image_size=args.image_size, val_split=args.val_split, test_spilt=args.test_split, seed=args.seed)

    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=os.cpu_count(),batch_size=args.batch_size, shuffle=True)
    val_dataloader   = DataLoader(val_dataset, pin_memory=True, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    test_dataloader   = DataLoader(test_dataset, pin_memory=True, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=True)

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
            


def build_scheduler(optimizer):
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min')
    
    return scheduler

  
def main():
    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # args
    parser = build_argparse()
    args = parser.parse_args()
    check_argparse(args)

    # data
    print('\n-------- Data Preparing --------\n')

    train_dataloader, val_dataloader, test_dataloader = build_train_val_test_dataset(args)

    print('\n-------- Data Preparing Done! --------\n')


    print('\n-------- Preparing Model --------\n')
    # model
    model = MetalModel(model_name = args.model_name, hidden_dim=args.hidden_dim, activation=args.activation)

    # pretrained model freeze
    if args.freeze:
        freeze_pretrain(model, True)
    else:
        freeze_pretrain(model, False)

    # loading previous model parameters or not
    if args.load_model_para:
        model.load_state_dict(torch.load('/home/rico-li/Job/Metal/model_save/'+args.load_model_para))
    else:
        pass

    # pass to CUDA device
    model = model.to(device)
    
    # TODO: need to discuss
    # criterion = WeightFocalLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.learning_rate, nesterov=True, weight_decay=0.01)
    scheduler = build_scheduler(optimizer)
            
    print('\n-------- Preparing Model Done! --------\n')

    # train
    print('\n-------- Starting Training --------\n')
    # prepare the tensorboard
    writer = SummaryWriter(f'runs/trial_{args.exp}')

    for epoch in range(args.epoch):
        start_time = time.time()

        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')

        # TRAINING LOOP
        print('---Training Loop begins---')
        read_time = time.time()
        
        # solving bact_size issue
        optimizer.zero_grad()

        for i, data in enumerate(train_dataloader, start=0):
            
            # --- test section---
            if args.test_section:
                if i == 1:
                    break
            # --- test section ---
             
            # move CUDA device
            input, target = data[0].to(device), data[1].to(device)

            
            output = model(input)

            loss = criterion(output, target)

            loss.backward()

            # solving bact_size issue
            if (i+1)%args.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_running_loss += loss.item()
            writer.add_scalar('Averaged loss', loss.item(), int(8511*0.6/args.batch_size)*epoch + i)
            
            print(
                f"[{epoch+1}, {i+1}]: %.3f" % (train_running_loss)
            )
            train_running_loss = 0.0
        
        lr = [group['lr'] for group in optimizer.param_groups]
        print('Epoch:', epoch+1,'LR:', lr[0])
        writer.add_scalar('Learning Rate', lr[0], epoch)

        print('---Training Loop ends---')
        print(f'---Training spend time: %.1f sec' % (time.time() - start_time))
        
        # VALIDATION LOOP
        with torch.no_grad():
            val_run_loss = 0.0
            print('\n---Validaion Loop begins---')
            batch_count = 0
            total_count = 0
            correct_count = 0
            for i, data in enumerate(val_dataloader, start=0):

                # --- test section---
                if args.test_section:
                    if i == 1:
                        break
                # --- test section ---

                input, target = data[0].to(device), data[1].to(device)

                output = model(input)
                loss = criterion(output, target)

                _, predicted = torch.max(output, 1)

                val_run_loss += loss.item()
                batch_count += 1
                total_count += target.size(0)
                

                correct_count += (predicted == target).sum().item()
            
            accuracy = (100 * correct_count/total_count)
            val_run_loss = val_run_loss/batch_count
            
            scheduler.step(val_run_loss)

            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)

            print(f"\nLoss of {epoch+1} epoch is %.3f" % (val_run_loss))
            print(f"Accuracy is %.2f %% \n" % (accuracy))
                
            print('---Validaion Loop ends---')
    writer.close()
    print('\n-------- End Training --------\n')
    

    print('\n-------- Saving Model --------\n')

    savepath = f'/home/rico-li/Job/Metal/model_save/{str(args.exp)}_{str(args.model_name)}.pth'
    torch.save(model.state_dict(), savepath)
        
    print('\n-------- Saved --------\n')
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
# python train.py --exp 19 --epoch 15 --model_name 'se_resnet152' --batch_size 16 --freeze True
# Unfreeze and load .pth
# python train.py --exp 19 --epoch 15 --model_name 'se_resnet152' --batch_size 16 --load_model_para 18_se_resnet152.pth --learning_rate 0.005