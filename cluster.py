import time
import os
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import cv2
from matplotlib.pyplot import figure

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

import pretrainedmodels

def featureExtractor(model_name = 'se_resnet152', hidden_dim=256, 
                    activation='relu', mode='train', batch_size = 64):
    assert mode == 'train', 'mode should be train mode'
    # TODO: combine activate here
    dataset = MetalDataset(mode=mode, transform=True, cluster_img=False, combine=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=os.cpu_count())
    
    oneDfea = torch.Tensor()
    total_lab = torch.Tensor().int()
    image_names = []

    print('--Using imagenet pretrain--')
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet') 
    model.eval()
    print('--model loaded--')

    for image, label, image_name in dataloader:
        features = model.features(image)
        features = features.view(features.size(0),-1)
        oneDfea = torch.cat((oneDfea, features), 0)
        total_lab = torch.cat((total_lab, label.int()), 0)
        image_names += image_name
        print(oneDfea.shape)
        print(len(image_names))

    torch.save(oneDfea, f'oneDfea_combine_True')
    torch.save(total_lab, f'oneDfea_lab_combine_True')

    f=open(f'oneDfea_name_combine_True.txt','w')
    image_names=map(lambda x:x+'\n', image_names)
    f.writelines(image_names)
    f.close()

    print('data saved')

def kmean(fea_path, label_path, name_path,num_clusters, mode, metal_train=False, save=True):
    oneDfea = torch.load(fea_path)
    feanamefile = open(name_path)
    feaName = feanamefile.read()
    feaName = feaName.splitlines()
    labels = torch.load(label_path)
    
    newlabel = torch.Tensor()
    img_name = []
    for i in torch.unique(labels):
        x = oneDfea[labels == i]
        index = [idx for idx, label in enumerate(labels.tolist()) if label == i]
        img_name += [feaName[i] for i in index]
        
        if num_clusters[i] != 0: 
            kmeans = KMeans(n_clusters=num_clusters[i], n_jobs = -1).fit(x)
            newlabel = torch.cat((newlabel, i*10 + torch.from_numpy(kmeans.labels_).float()))
            inertia = kmeans
        else:
            newlabel = torch.cat((newlabel, torch.tensor(([i*10]*x.shape[0]), dtype=torch.float)))
            print(f'class does not do the kmean: {i}')
            print(f'number of samples: {x.shape[0]}')

        print(f'\n-- class {i} is done --\n')
    
    if save:
        torch.save(newlabel, f'oneDfea_newlab_1113merge')
        f = open(f'oneDfea_imgname_1113merge.txt','w')
        img_name=map(lambda x:x+'\n', img_name)
        f.writelines(img_name)
        f.close()
    else:
        pass
    
    return newlabel

def clusterNumber(fea_path, label_path, list_num_clusters, class_i_list):
    oneDfea = torch.load(fea_path)
    labels = torch.load(label_path)

    for class_i in class_i_list:

        x = oneDfea[labels == class_i]
        print('\n number of samples in the class: ', x.shape[0])

        inertia = []
        silhouette_score = []
        for num_clusters in list_num_clusters:
            kmeans = KMeans(n_clusters=num_clusters, n_jobs = -1).fit(x)
            params = kmeans.get_params()
            label = kmeans.labels_

            if num_clusters == 1:
                inertia += [kmeans.inertia_]
            else:
                silhouette_score += [metrics.silhouette_score(x, label, metric='euclidean')]
                inertia += [kmeans.inertia_]

        index = silhouette_score.index(max(silhouette_score))
        knumber = list_num_clusters[index]

        fig, (ax1, ax2) = plt.subplots(1, 2)
        # elbow method
        ax1.scatter(list_num_clusters,inertia)
        ax1.set_title('elbow')
        ax1.set(xlabel = 'k number', ylabel = 'wss')
        ax1.label_outer()
        # silhouette_score
        ax2.scatter(list_num_clusters[1:], silhouette_score)
        ax2.set_title('silhouette_score')
        ax2.set(xlabel = 'k number', ylabel = 'silhouette_score')
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        plt.savefig(f'k_number_c{class_i}')

        print(params)

    return knumber

def nakeEyesCheck(newlab_path, name_path, show_id):
    newlabel = torch.load(newlab_path)
    feanamefile = open(name_path)
    feaName = feanamefile.read()
    feaName = feaName.splitlines()

    newlabel = newlabel.numpy()
    feaName  = np.array(feaName)

    print(newlabel[0:50])

    print(len(np.where(newlabel == 0)[0]))
    print(len(np.where(newlabel == 1)[0]))
    print(len(np.where(newlabel == 2)[0]))

    class0Name = feaName[np.where(newlabel == 0)][show_id]
    class1Name = feaName[np.where(newlabel == 1)][show_id]
    class2Name = feaName[np.where(newlabel == 2)][show_id]

    img1 = cv2.imread(f'/home/rico-li/Job/Metal/Image/GB/{class0Name}', cv2.IMREAD_COLOR)
    img2 = cv2.imread(f'/home/rico-li/Job/Metal/Image/GB/{class1Name}', cv2.IMREAD_COLOR)
    img3 = cv2.imread(f'/home/rico-li/Job/Metal/Image/GB/{class2Name}', cv2.IMREAD_COLOR)
    img  = np.concatenate((img1, img2, img3), axis=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    figure(figsize=(20, 10))

    plt.imshow(img)
    plt.title('subclass examples')
    plt.axis('off')
    plt.show()


def visual_kmean_pca(fea_path, old_label_path, new_label_path=None, 
                    name_path=None,label_i=None, n_components=2, all=False):
    oneDfea = torch.load(fea_path)
    labels = torch.load(old_label_path)
    feanamefile = open(name_path)
    feaNames = feanamefile.read()
    feaNames = feaNames.splitlines()
    feaNames = np.array(feaNames)

    if not all:
        newlabels = torch.load(new_label_path)
        index = newlabels//10 == label_i

        newlabel = newlabels[index]
        feaName  = feaNames[np.where(index.numpy())]
        label = newlabel.numpy()
        pcashow = oneDfea[labels == label_i]
        colors = ['blue','red','green','purple']
    else:
        label = labels
        pcashow = oneDfea
        colors = ['blue']

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(pcashow)    
    pca_one = pca_result[:,0]
    pca_two = pca_result[:,1]

    plt.figure(figsize=(8,8))
    plt.scatter(pca_one, pca_two, c=label, cmap=matplotlib.colors.ListedColormap(colors))
    if not all:
        for i, txt in enumerate(feaName):
            if i%15 == 14:
                plt.annotate(txt, (pca_one[i]-100, pca_two[i]))
    else:
        pass

    plt.show()


if __name__ == '__main__':
    with torch.no_grad():
        # --- feature extraction ---
        # featureExtractor()
        
        # --- kmean ---
        start_time = time.time()
        #              [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        # k_num_list = [3,3,2,4,4,4,2,2,3,2, 0, 4, 0, 0, 0]
        # TODO: Combining class 11 and 13 to class
        #                [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        # k_num_list   = [3,3,2,4,4,4,2,2,3,2, 0, 4, 0, 0]

        # newlabel = kmean('oneDfea_combine_True', 'oneDfea_lab_combine_True', 'oneDfea_name_combine_True.txt',
        #                 k_num_list, mode='train')
        # print(newlabel.shape)
        # print(f'time: %.2f' % (time.time() - start_time))
        
        # cluster
        # start_time = time.time()
        # knumber = clusterNumber('oneDfea_train_metal_trained_False', 'oneDfea_lab_train_metal_trained_False', [1,2,3,4,5,6,7], 
        #                         class_i_list=[i for i in range(15)])
        # print(knumber)
        # print(f'time: %.2f' % (time.time() - start_time))


        # PCA
        visual_kmean_pca('oneDfea_train_metal_trained_False', 'oneDfea_lab_train_metal_trained_False', 
                          'oneDfea_newlab_train_metal_train_False_13changed', 'oneDfea_name_train_metal_trained_False.txt', 
                          n_components=2, label_i=9)

        # --- nake eyes verify
        # nakeEyesCheck('oneDfea_newlab_train', 'oneDfea_name_train.txt', 0)