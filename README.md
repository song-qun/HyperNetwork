# HyperNetwork

This is the repo for the EWSN'22 paper "Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge"

## Getting started

pip3 install -r requirements.txt

## Download datasets

### MNIST

Download [MNIST dataset](http://yann.lecun.com/exdb/mnist/) and store it under folder data_m/. Organize the data by following hierachy.

```
/data_m
  /MNIST
    /processed
      test.pt
      training.pt
    /raw
      ...
```

### notMNIST
      
Download [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) and store it under data_nm/. Organize the data by following hierachy.

```
/data_nm
  /Test
    /A
    /B
    ...
    /J
  /Train
    /A
    /B
    ...
    /J
```
    
### GTSRB

Download [GTSRB dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) in which the images are resized to 32x32. Organize the data by following hierachy.

```
/traffic-signs-data
  test.p
  train.p
  valid.p
```

### KUL

Download [KUL BelgiumTS dataset](https://btsd.ethz.ch/shareddata/). Preprocess the downloaded data with KUL_preprocess.py. Orgainize the data by following hierachy.

```
/KUL
  test_data.npy
  test_labels.npy
  train_data.npy
  train_labels.npy
```

## Train HyperNet

python3 train_hypernet.py --cuda --dataset mnist

python3 train_hypernet.py --cuda --dataset gtsrb

python3 train_hypernet.py --cuda --dataset kul

## Generate experiment results in the paper

python3 experiments.py --cuda
