import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import MetalModel
from dataset import MetalDataset
from utils import cluster2target

def confusionMatrix(model_path, model_name, mode, cluster_img, 
                    output_class, plotclass, cluster_img_val=False, merge=True):
    model = MetalModel(model_name = model_name, hidden_dim=256, 
                        activation='relu', output_class=output_class)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model loaded')
    model = model.cuda()
    
    with torch.no_grad():
        dataset = MetalDataset(mode=mode, cluster_img=cluster_img)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, 
                                pin_memory=True, num_workers=2*os.cpu_count())

        targets = torch.Tensor().type(torch.long)
        predicts = torch.Tensor().type(torch.long).cuda()
        for i, data in enumerate(dataloader, start=0):
            input, target = data[0].cuda(), data[1]
            # TODO: merge case only
            if merge:
                target[target == 13] = 11
                target[target == 14] = 13
            else:
                pass
            targets = torch.cat((targets, target))
            output = model(input) 
            _, predicted = torch.max(output, 1)
            if cluster_img_val:
                predicted = cluster2target(predicted)
            else:
                pass
            predicts = torch.cat((predicts, predicted))
            print(f'-- {i} batch--')

        correct_count = (predicts == targets.cuda()).sum().item()
        accuracy = (100 * correct_count/len(dataset))
        print(f'\n Accuracy on {mode} set: %.2f %% \n' % (accuracy) )
        targets = targets.numpy()
        predicts = predicts.cpu().numpy()
        c_matrix = confusion_matrix(targets, predicts, normalize='true',
                                    labels=[i for i in range(plotclass)])
    return c_matrix



if __name__ == '__main__':
    start_time = time.time()
    mode = 'test'
    output_class = 36
    plotclass = 14
    cluster_img = False
    cluster_img_val = True
    merge = True

    local_path = '/home/rico-li/Job/Metal/model_save/69_se_resnext101_32x4d.pth'
    server_path = '/home/aiuser/Job/MetalClassification/mode_save/69_se_resnext101_32x4d.pth'
    
    if os.getcwd() == '/home/rico-li/Job/Metal':
        path = local_path
    else:
        path = server_path

    c_matrix = confusionMatrix(model_path=path, 
                    model_name='se_resnext101_32x4d', mode=mode, 
                    cluster_img = cluster_img, output_class=output_class, 
                    plotclass=plotclass, merge=merge, cluster_img_val=cluster_img_val)
    
    import matplotlib.pyplot as plt

    figure = plt.figure()
    axes = figure.add_subplot(111)     

    axes.matshow(c_matrix)
    axes.set_title(f'Confusion Matrix: {mode} set')
    axes.set(xlabel = 'Predicted',ylabel = 'Truth')
    axes.set_xticks(np.arange(0, plotclass-1))
    axes.set_yticks(np.arange(0, plotclass-1))
    caxes = axes.matshow(c_matrix, interpolation ='nearest') 
    figure.colorbar(caxes) 
    print(f'--- %.1f sec ---' % (time.time() - start_time))
    plt.show() 
