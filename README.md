# Work Log
## TODO:
1. auto-encoder + anomaly detection
2. Ensemble
3. Use domain knowledage
4. [Long tail classification](http://bangqu.com/2gQa9r.html?fbclid=IwAR3HqmMLyVOeEz0fq3hWVZFtjUEw9AWRIBpZgZy35a8ruappRb4gP4wihfc)
5. metric learning
6. self labeling
7. [Self-supervise learning](https://blog.voidful.tech/paper%20reading/2020/06/28/paper-notes-simclr/?fbclid=IwAR3JN75zoG74O1J-1x2lo4rKrTbF7G0Obsvfb5zp-72J33RG1_2dzytPyZI)
8. [Early Stop](https://github.com/Bjarten/early-stopping-pytorch)


## Ideas
* direct image PCA, seperate minor classes.
* trian the A model with 13 classes (w/o 11 & 13)
* then trian the B model with only 11 and 13.


## IMPORANT UPDATED (7/8)
add in `model.train()` and `model.eval()` in the train.py
## 7/8
#### Test run 3
use trial 68 as freeze
    
    trial 69
    unfreeze
    :::info
    epoch 20
    91.34 %
    2621.9 sec
    :::

#### Test run 2
re-tain trial 64 (73.56 %) with fine-tuning

    trial 68
    freeze
    :::info
    epoch 6
    --lr_name 'StepLR' --optim Adam
    73.87 %
    :::

    trial 69
    freeze
    :::info
    epoch 6
    --lr_name 'StepLR' --optim SGD
    73.29 %
    701.7 sec
    :::

#### Test run 1
re-train trial 65 (88.84 %) with fine-tuning
    trial 66 
    :::info
    epoch 6
    --lr_name 'StepLR' --optim SGD
    89.15 %
    787.5 sec
    :::

    trial 67
    :::info
    epoch 6
    --lr_name 'StepLR' --optim Adam
    89.89 %
    790.0 sec
    :::

## 7/7
#### Test run 2
* using the cluster label
* file `oneDfea_combine_True` is class 11 and 13 merge feature maps
* file `oneDfea_lab_combine_True.txt` is corresponded image names
* file `oneDfea_newlab_1113merge` is class 11 and 13 merge cluster new labels
* file `oneDfea_imgname_1113merge.txt` is corresponded image names
* file `oneDfea_train_label36` is the continuous labels
    
    trial 64
    freeze
    :::info
    epoch 10
    73.56 % 1189.4 sec
    :::

    trial 65
    unfreeze
    :::info
    epoch 15
    88.84 %
    1949.2 sec
    :::

#### Test run 1
* train freeze model (trial 56) more

    trial 59
    freeze
    :::info
    epoch 5
    76.18 %
    overfiting at the last epoch
    :::

    trial 60 
    freeze
    :::info
    StepLR
    epoch 5
    76.18 %
    :::
train more have minor enhance

#### trial 58 check
cluster image will improve the performace

## 7/6
* It is not an good idea using fc in the middle of auto-encoder (auto_encoder_trial_15)
* Changing the feature number in default auto-encoder model (trial 16 5 epochs)
    * check the loss threshold of class 10 (cannot get with this model)
* Refine model: add batch normalization (trial 17)
* TODO: Postone for now

#### Test run 1
* let class 11 and 13 to be the same class.
    * use class label with range (0~13), original 11 & 13 merge to class 11.
    * class 14 change to class 13
* Discard cluster class
    * model output change
    * dataset label change
    * train.py val loop change

    trial 56
    freeze
    :::info
    epoch 10
    75.01 %
    1177.3 sec
    :::

    trial 57
    unfreeze
    :::info
    epoch 15
    89.31 %
    1939.0 sec
    :::

    trial 58
    re-train trial 57
    :::info
    StepLR, SGD
    5 epoch
    89.54 %
    :::


## 7/2
label 13 class is the worst in trial 47
try to add cluster method to label13 
-> guess not, cluster seperation is very bad.

#### Test run 1
auto-encoder
1. Default auto-encoder model (v)


## 7/1
#### Test run 2
1. Use se_resnext101_32x4d model
    trial 45
    :::info
    epoch 1
    59.42 %
    690.6 sec
    :::

2. Using freeze-unfreeze
    trial 46
    freeze
    :::info
    epoch 10
    69.68 %
    1182.7 sec
    epoch 7 of val loss shows the sign of overfitting
    :::

    trial 47
    unfreeze
    :::info
    epoch 15
    90.21 %
    1968.7 sec
    while lr 0.0001 performance increase
    :::

    trial 48
    unfreeze and re-train trial 47 with small lr
    and using SGD optim
    :::info
    epoch 5
    90.80 %
    659.4 sec
    :::

    trial 49
    unfreeze and re-train trial 47 with small lr (StepLR, step_size=2, gamma=0.1) and using SGD optim
    :::info
    epoch 5
    90.80 %
    651.5 sec
    :::

----------- Correct Accuracy Updated until Here-----------


#### Test run 1
1. use the SGD+momentum re-run trial 43
    trial 43
    :::info
    epoch 5
    82.8 %
    635.6 sec
    the overfitting sign gone
    :::
2. use the SGD+momentum re-run trial 43
    trial 44
    :::info
    epoch 10
    81.32 %
    no improvement
    :::

## 6/30
#### Test run 3
!!!!!!!!!! IMPORTANT !!!!!!!!!!!
YOU have not use AUG in train set (V)
    
    trial 42
    freeze
    :::info
    epoch 15, lr fixed
    60 %
    1770.4 sec
    :::

    trial 43 (file get washed @@)
    unfreeze
    :::info
    epoch 20
    82.6 %
    sign of overfitting around epoch 13
    :::



#### Test run 2.
Using freeze-unfreeze method
    
    trial 39
    freeze
    :::info
    epoch 15
    66.27 %
    1670.6 sec
    :::

    trial 40
    unfreeze and ReduceLROnPlateau with patience=4
    :::info
    epoch 20
    81.0 %
    2508.8 sec
    sign of overfitting
    :::

#### Test run 1
* cluster2target fixed
    *  val loss bug fixed

SGD -> Adam, default parameters
    trial 37
    :::info
    epoch 5
    67.02 %
    406.9 sec
    :::

    trial 38
    :::info
    epoch 15
    75 %
    1897.9 sec
    at epoch 6, sign of overfitting
    :::

## 6/29
#### Find the best number of clusters (V)
k_numbers = [3,3,2,4,4,4,2,2,3,2,0,4,0,0,0]
#### Modify dataset.py (V)
#### Modify train.py (V)

#### Test run 1 (trial 28)
1. 37 subclasses 
    :::info
    3 epochs
    26.87 %
    :::
#### Test run 2 (trial 29 & trial 30)
1. SGD -> Adam, default parameters
    :::info
    3 epochs 
    27.50 %
    :::
    :::info
    15 epochs
    train loss is underfitting
    val loss is overfitting
    sth is wrong @@
    check the label!!
    :::


## 6/24
#### Create own cluster dataset
1. features from trial 23
    1. Feature separation is not good.
    2. kmean does not seperate the class well.
2. features from imagenet (V)
    1. well seperate!
    2. kmean can have good seperations


## 6/23
#### Test run 1 (trial 25)
model_name change to se_resnet152 for default
batch_size 16 as default
1. Using the cluster dataset
    1. Directly use the 66 subclass
    :::info
    --epoch 10 -model_name 'se_resnet152' --batch_size 16
    50.08 %
    :::


## 6/22
#### Test run 1 (trial 22 & 23)
1. Using class distribution train/val/test set
2. Pretrain trial:
    Freeze
    :::info 
    --epoch 15 --model_name 'se_resnet152' --batch_size 16
    64.1 %
    1645.0 sec
    :::

    Unfreeze
    :::info 
    --epoch 20 --model_name 'se_resnet152' --batch_size 16
    81.36 %
    3100.4 sec
    :::


## 6/20
#### Test run 1 (trial 18 & 19<-file get messup @@)
1. Pretrainmodel parameter freeze
    1. freeze pretrain model parameter, using default lr
    :::info
    trial 18: 
    lr = 0.01
    --epoch 10 --model_name 'se_resnet152' --batch_size 16
    59.07 %
    1120.5 sec
    :::

    2. Loading the previous model path, then unfreeze the model, and change the lr to smaller one.
    :::info
    trial 19:
    lr = 0.005
    --epoch 10 --model_name 'se_resnet152' --batch_size 16
    87.5 %
    1258.5 sec 
    :::


## 6/19
#### Test run 1 (trial 17)
1. CrossEntropy -> WeightFocalLoss
:::info
'se_resnet50'
batch_size 32
epoch 12 30.1 %
:::




## 6/18
#### Test run 1 (file missing)
1. StepLR(step_size=3, gamma=0.1)
2. model_name 'se_resnet50'
:::info
batch_size 32
epoch 12 72.2 %
1374.3 sec
:::

#### Test run 2 (trial 13)
1. StepLR -> ReduceLROnPlateau
2. solving batch_size issue
3. model_name 'efficientnet-b7'
:::info
batch_size = 5
epoch 1 12.14 % 
246.9 sec
:::

#### Test run 3 (trial 15)
1. StepLR -> ReduceLROnPlateau
2. solving batch_size issue
3. model_name 'se_resnet152'
:::info
batch_size = 16
epoch 10 77.67 %
1276.7 sec
:::


## 6/17
### Hyparameter Tuning
Note that below is add up modifications

#### Test run 1
1. optimizer: add momentum=0.9, nesterov=True
:::info
1 epoch: 45.4 %
:::

#### Test run 2
1. learning rate: LambdaLR -> StepLR(step_size=1, gamma=0.1)
2. optimizer: add weight_decay=0.01
:::info
Trial 7
1 epoch: 42.1 %
2 epoch: 47.4 %
3011.4 sec
:::

#### Speed problem fixing
1. rewrite dataset
2. num_workers
3. pin_memory
:::info
1/8 time saved
::::

### Test run 3
1. learning rate: LambdaLR -> StepLR(step_size=1, gamma=0.1)
2. optimizer: add weight_decay=0.01
:::info
Trial 7
20 epoch: 48 %
2375.6 sec
:::



## 6/16
In T2H folder
image: 
T1H_Q1558-1090326175042092.jpg
T1H_Q1558-1090326175049966.jpg
T1H_Q1558-1090326175101066.jpg

I move them to T1H folder.
:::info
accuracy baseline: 56.7% (trial_3)
:::
> file is saved as trial_3

