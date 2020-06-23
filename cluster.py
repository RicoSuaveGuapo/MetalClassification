import time

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns   

import torch
from model import MetalModel
from dataset import MetalDataset
from torch.utils.data import DataLoader

from kmeans_pytorch import kmeans, kmeans_predict


def featureExtractor(model_name = 'se_resnet152', hidden_dim=256, activation='relu', mode='train', bacth_size = 20):

    model = MetalModel(model_name = model_name, hidden_dim=hidden_dim, activation=activation)
    model.load_state_dict(torch.load('/home/rico-li/Job/Metal/model_save/23_se_resnet152.pth'))
    print('load model success')

    assert mode in ['train','val','test']
    dataset = MetalDataset(mode=mode, transform=True)
    batch_size = bacth_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    oneDfea = torch.Tensor()
    total_lab = torch.Tensor().int()
    image_name = []
    for i, (image, label, image_name) in enumerate(dataloader):
        features = model.features(image)
        features = features.view(features.size(0),-1)
        oneDfea = torch.cat((oneDfea, features), 0)
        total_lab = torch.cat((total_lab, label.int()), 0)
        image_name += image_name
        print(oneDfea.shape)
        # break

    torch.save(oneDfea, f'oneDfea_{mode}')
    torch.save(total_lab, f'oneDfea_lab_{mode}')

    f=open(f'oneDfea_name_{mode}.txt','w')
    image_name=map(lambda x:x+'\n', image_name)
    f.writelines(image_name)
    f.close()

    print('data saved')

def kmean(fea_path, label_path, fea_name_path, num_clusters):
    oneDfea = torch.load(fea_path)
    labels = torch.load(label_path)
    feanamefile = open(fea_name_path)
    feaName = feanamefile.read()
    feaName = feaName.splitlines()

    newlabel = torch.Tensor()
    all_kmean = []
    for i in torch.unique(labels):
        x = oneDfea[labels == i]
        if x.shape[0] > 200:
            kmeans = KMeans(n_clusters=num_clusters).fit(x) # TODO: [TBD]
            all_kmean += kmeans
            newlabel = torch.cat((newlabel, i*10 + torch.from_numpy(kmeans.labels_)))
        else:
            newlabel = torch.cat((newlabel, i*10))
    
    return all_kmean, newlabel, feaName

def visual_kmean_pca(fea_path, label_path, fea_name_path, label_i, n_components=2):
    oneDfea = torch.load(fea_path)
    labels = torch.load(label_path)
    feanamefile = open(fea_name_path)
    feaName = feanamefile.read()
    feaName = feaName.splitlines()

    pcashow = oneDfea[labels == label_i]

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(pcashow)
    
    pca_one = pca_result[:,0]
    pca_two = pca_result[:,1]
    colors = ['blue','red','purple']

    plt.figure(figsize=(8,8))
    plt.scatter(pca_one, pca_two, c=newlabel, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()


if __name__ == '__main__':
    with torch.no_grad():
        # --- feature extraction ---
        featureExtractor(model_name = 'se_resnet152', hidden_dim=256, activation='relu', mode='train', bacth_size = 20)
        
        # -- kmean ---
        # all_kmean, newlabel, feaName = kmean('oneDfea_train','oneDfea_lab_train','oneDfea_name_train.txt', 3)
        
       

        # prediction
        # kmeans.predict()



        # PYTORCH-KMEAN
        # if torch.cuda.is_available():
        #     device = torch.device('cuda:0')
        
        # # cluster_ids = torch.Tensor()

        # for i in torch.unique(labels):
        #     x = oneDfea[labels == i]
        #     if x.shape[0] > 200:
        #         cluster_ids_x, cluster_centers = kmeans(X=x, num_clusters=num_clusters, distance='euclidean', device=device)
        #         break
        #     else:
        #         continue
        # print(cluster_ids_x)
        