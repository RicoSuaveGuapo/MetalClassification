# Metal_Classification

# Work Log

## TODO
1. class distribution into the train/val/test set.
2. Adam then SGD+momentum for fine-tuning
3. Early Stop
4. Increase the amount of epoch of freeze model
5. Using the cluster dataset

## 6/20
#### Test run 1 (trial 18 & 19)
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
