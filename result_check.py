import time
import os
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
    model = model.cuda()
    
    with torch.no_grad():
        dataset = MetalDataset(mode=mode)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=2*os.cpu_count())

        targets = torch.Tensor().type(torch.long)
        predicts = torch.Tensor().type(torch.long).cuda()
        for i, data in enumerate(dataloader, start=0):
            input, target = data[0].cuda(), data[1]
            targets = torch.cat((targets, target))
            output = model(input) # (37)
            _, predicted = torch.max(output, 1) # (37)
            predicted = cluster2target(predicted) # (15)
            predicts = torch.cat((predicts, predicted))
            print(f'-- {i} batch--')
            # if i == 0:
            #     break

        targets = targets.numpy()
        predicts = predicts.cpu().numpy()
        c_matrix = confusion_matrix(targets, predicts, normalize='true', labels=[i for i in range(15)])
    return c_matrix



if __name__ == '__main__':
    start_time = time.time()
    mode = 'test'
    c_matrix = confusionMatrix(model_path='/home/rico-li/Job/Metal/model_save/47_se_resnext101_32x4d.pth', 
                    model_name='se_resnext101_32x4d', mode=mode)
    
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(111)     

    axes.matshow(c_matrix)
    axes.set_title(f'Confusion Matrix: {mode} set')
    axes.set(xlabel = 'Predicted',ylabel = 'Truth')
    axes.set_xticks(np.arange(0, 14))
    axes.set_yticks(np.arange(0, 14))
    caxes = axes.matshow(c_matrix, interpolation ='nearest') 
    figure.colorbar(caxes) 
    print(f'--- %.1f sec ---' % (time.time() - start_time))
    plt.show() 
