# Metal_Classification

# Work Log

## TODO
1. Using the cluster dataset w/o synthetic data (V)
2. Create own cluster dataset 
3. PCA or t-SNE for feature maps to 2D, to check whether clustering or not.
2. Adam then SGD+momentum for fine-tuning
3. [Early Stop](https://github.com/Bjarten/early-stopping-pytorch)
4. Mixup
5. [Long tail classification](http://bangqu.com/2gQa9r.html?fbclid=IwAR3HqmMLyVOeEz0fq3hWVZFtjUEw9AWRIBpZgZy35a8ruappRb4gP4wihfc)

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
2. Create own cluster dataset (using the result from trial 23)

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

# MetalClassification
