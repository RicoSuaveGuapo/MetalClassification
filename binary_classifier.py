import time
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import cv2
from matplotlib.pyplot import figure
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter

from model import MetalModel
from dataset import BinaryDataset
from torch.utils.data import DataLoader
from train import freeze_pretrain

def featurePCA(feature_path, label_path, mode:str, sample_index_path=None):
    assert mode in ['all','cluster']

    labels = torch.load(label_path)
    colors = ['red', 'blue']

    features = torch.load(feature_path)
    if mode == 'cluster':
        idx = torch.load(sample_index_path)
        features = features[idx]
        legend_label = ['cluster 0', 'cluster 1']
        relabels = labels
    else:
        legend_label = ['11 and 13','others']
    
        # re-labeling
        relabels = torch.Tensor([]).long()
        for i, label in enumerate(labels):
            if (label == 11) or (label == 13):
                relabel = torch.tensor([0])
                relabels = torch.cat((relabels, relabel))
            else:
                relabel = torch.tensor([1])
                relabels = torch.cat((relabels, relabel))

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)    
    pca_one = pca_result[:,0]
    pca_two = pca_result[:,1]

    plt.figure(figsize=(8,8))
    scatter = plt.scatter(pca_one, pca_two, c=relabels, cmap=matplotlib.colors.ListedColormap(colors), 
                alpha=0.3, edgecolors=None)    
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_label)
    plt.title('Feature map PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()

def kmeanClassifier(feature_path, label_path, mode:str, model_path=None, sample_factor=0.1):
    assert mode in ['train','inference']
    
    if mode == 'train':
        # training set
        features = torch.load(feature_path)
        torch.manual_seed(42)
        idx = torch.randint(low=0, high=features.size(0), size=(int(features.size(0)*sample_factor),))
        sample_features = features[idx]
        # labels = torch.load(label_path)

        kmeans = KMeans(n_clusters=2, n_jobs = -1).fit(sample_features)
        newlabel = torch.tensor(kmeans.labels_, dtype=torch.long)

        torch.save(newlabel, 'binary_label')
        print('new label saved')

        torch.save(idx, 'sample_index')
        print('new label sampled index saved')

        
        # Save to file in the current working directory
        kmeans_name = "kmeans.pkl"
        with open(kmeans_name, 'wb') as file:
            pickle.dump(kmeans, file)
        print('model saved')
    else:
        features = torch.load(feature_path)
        labels = torch.load(label_path)

        # re-labeling: 11 & 13 -> 0, others -> 1
        relabels = torch.Tensor([]).long()
        for i, label in enumerate(labels):
            if (label == 11) or (label == 13):
                relabel = torch.tensor([0])
                relabels = torch.cat((relabels, relabel))
            else:
                relabel = torch.tensor([1])
                relabels = torch.cat((relabels, relabel))

        # Load model
        with open(model_path, 'rb') as file:
            kmeans = pickle.load(file)
        
        # Calculate the accuracy score
        Xtest  = features # tensor
        Ypredict = kmeans.predict(Xtest) # numpy
        Ypredict = torch.from_numpy(Ypredict)
        print(relabels[:20])
        print(Ypredict[:20])

        correct_count = (Ypredict == relabels).sum().item()
        score = 100 * correct_count/(relabels.shape[0])
        print("\nInference accuracy: {0:.2f} %\n".format(score))

def CNNClassifier(exp, epochs, model_name='resnet18', batch_size=16, task='1113others',freeze=False, load_model_para=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    model = MetalModel(model_name, 256, output_class=2)
    model = model.to(device)

    train_dataset = BinaryDataset(mode='train', transform=True, 
                            image_size=256, val_split=0.3, 
                            test_spilt=0.1, seed=42, task=task)
    train_dataloader = DataLoader(train_dataset, pin_memory=True, 
                            num_workers=os.cpu_count(), 
                            batch_size=batch_size, shuffle=True)

    val_dataset = BinaryDataset(mode='val', transform=True, 
                            image_size=256, val_split=0.3, 
                            test_spilt=0.1, seed=42, task=task)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, 
                            num_workers=2*os.cpu_count(), 
                            batch_size=batch_size, shuffle=True)

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=0.001, nesterov=True, weight_decay=0.01)
    weighting = torch.tensor(train_dataset.classWeight()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weighting)
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', patience=1)
    # scheduler = StepLR(optimizer, step_size=4, gamma=0.1)

    if freeze:
        print('model freeze')
        freeze_pretrain(model, True)
    else:
        print('model unfreeze')
        freeze_pretrain(model, False)

    if load_model_para:
        model.load_state_dict(torch.load(f'/home/rico-li/Job/Metal/model_save/{load_model_para}'))
        print('model loaded')
    else:
        pass

    writer = SummaryWriter(f'runs/CNNClassifier1113_trial_{exp}')


    for epoch in range(epochs):
        start_time = time.time()

        train_running_loss = 0.0
        print(f'\n---The {epoch+1}-th epoch---\n')
        print('[Epoch, Batch] : Loss')
        #  --------------------------- TRAINING LOOP ---------------------------
        print('---Training Loop begins---')
        optimizer.zero_grad() 
        model.train()
        for i, data in enumerate(train_dataloader, start=0):
            input, target = data[0].to(device), data[1].to(device)
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            train_running_loss += loss.item()
            if (i+1)%batch_size == 0: 
                writer.add_scalar('Batch-Averaged loss', train_running_loss/(batch_size), batch_size*epoch + int((i+1)/(batch_size)))
                print( f"[{epoch+1}, {int((i+1)/(batch_size))}]: %.3f" % (train_running_loss/(batch_size)) )
                optimizer.step()
                optimizer.zero_grad()
                train_running_loss = 0.0
        
        lr = [group['lr'] for group in optimizer.param_groups]
        print('Epoch:', f'{epoch+1}/{epochs}',' LR:', lr[0])
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
            predicts = torch.Tensor().type(torch.long).cuda()
            targets = torch.Tensor().type(torch.long).cuda()
            for i, data in enumerate(val_dataloader, start=0):
                input, target = data[0].to(device), data[1].to(device)
                targets = torch.cat((targets, target))
                output = model(input)
                _, predicted = torch.max(output, 1)
                loss = criterion(output, target)
                val_run_loss += loss.item()
                predicts = torch.cat((predicts, predicted))
                batch_count += 1
        
            correct_count = (predicts == targets).sum().item()
            accuracy = (100 * correct_count/len(val_dataset))
            val_run_loss = val_run_loss/batch_count
            scheduler.step(val_run_loss)
            # scheduler.step()
            writer.add_scalar('Validation accuracy', accuracy, epoch)
            writer.add_scalar('Validation loss', val_run_loss, epoch)
            print(f"\nLoss of {epoch+1} epoch is %.3f" % (val_run_loss))
            print(f"Accuracy is %.2f %% \n" % (accuracy))
            print('---Validaion Loop ends---')
            print(f'---Validaion spend time: %.1f sec' % (time.time() - start_time))
    writer.close()

    print('\n-------- Saving Model --------\n')
    savepath = f'/home/rico-li/Job/Metal/model_save/CNNClassifier1113_{model_name}_{exp}.pth'
    torch.save(model.state_dict(), savepath)
    print('-------- Saved --------')


if __name__ == '__main__':
    start_time = time.time()
    # kmeanClassifier('oneDfea_train_metal_trained_False', 
    #                 'oneDfea_lab_train_metal_trained_False', model_path="kmeans.pkl", mode='inference', sample_factor=0.3) 
    #                                                                         # factor > 0.5 will OOM
    # print(f'=== kmeans spends : %.2f sec ===' % (time.time() - start_time))

    # featurePCA('oneDfea_train_metal_trained_False', 'binary_label', 
    #                 mode='cluster', sample_index_path='sample_index')
    CNNClassifier(exp=9, epochs=20, model_name='resnet152', batch_size=16, freeze=False, task='1113')
    print(f'\n=== CNNClassifier spends : %.2f sec ===\n' % (time.time() - start_time))
    pass