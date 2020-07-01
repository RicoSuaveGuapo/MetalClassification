import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import MetalModel
from dataset import MetalDataset
from utils import cluster2target

def confusionMatrix(model_path, model_name, mode):
    model = MetalModel(model_name = model_name, hidden_dim=256, activation='relu', cluster_img=True)
    model.load_state_dict(torch.load(model_path))

    dataset = MetalDataset(mode=mode)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    for i, data in enumerate(dataloader, start=0):
        input, target = data[0], data[1]
        target = target.numpy()
        output = model(input) # (37)
        _, predicted = torch.max(output, 1) # (37)
        
        # cluster label to target label
        predicted = cluster2target(predicted) # (15)
        predicted = predicted.cpu().numpy()
        print(target)
        print(predicted)
        c_matrix = confusion_matrix(target, predicted, normalize='true')
        break
    return c_matrix



if __name__ == '__main__':
    c_matrix = confusionMatrix(model_path='/home/rico-li/Job/Metal/model_save/47_se_resnext101_32x4d.pth', 
                    model_name='se_resnext101_32x4d', mode='val')
    print(c_matrix)