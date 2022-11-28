import torch
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
# from scipy.misc import imread
from torch import Tensor
import pickle
from torch.utils.data import DataLoader
import utils


def load_mnist(args):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_m/'
    # path = 'mnist_FADA/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader


# def mnist_dataloader(batch_size=256,train=True):

#     dataloader=DataLoader(
#     datasets.MNIST('data_m/',train=train,download=False,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5,),(0.5,))
#                    ])),
#     batch_size=batch_size,shuffle=True)

#     return dataloader


def load_notmnist(args):
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    path = 'data_nm/'
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader

def load_fashion_mnist(args):
    path = 'data_f'
    torch.cuda.manual_seed(1)
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                batch_size=100, shuffle=False, **kwargs)
    return train_loader, test_loader


def load_cifar(args):
    path = 'data_c/'
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
            shuffle=False, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_cifar_hidden(args, c_idx=[0,1,2,3,4]):
    path = './data_c'
    if args.scratch:
        path = '/scratch/eecs-share/ratzlafn/' + path
    kwargs = {'num_workers': 2, 'pin_memory': True, 'drop_last': True}
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  
    def get_classes(target, labels):
        label_indices = []
        for i in range(len(target)):
            if target[i][1] in labels:
                label_indices.append(i)
        return label_indices

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
            download=False, transform=transform_train)
    train_hidden = torch.utils.data.Subset(trainset, get_classes(trainset, c_idx))
    trainloader = torch.utils.data.DataLoader(train_hidden, batch_size=32,
            shuffle=True, **kwargs)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
            download=False, transform=transform_test)
    test_hidden = torch.utils.data.Subset(testset, get_classes(testset, c_idx))
    testloader = torch.utils.data.DataLoader(test_hidden, batch_size=100,
            shuffle=False, **kwargs)
    return trainloader, testloader


def load_batch(fpath):
    images = []
    labels = []
    num_classes = 43
    with open(fpath, 'rb') as rfile:
        train_dataset =  pickle.load(rfile)
    for image in train_dataset['features']:
        # print(image.min(),image.max())
        images.append((image/255)-.5)
    for label in train_dataset['labels']:
        # labels.append(np.eye(num_classes)[label])
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


class readGTSRB:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data

        img, lab = load_batch('./traffic-signs-data/train.p')
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data,dtype=np.float32)
        self.train_labels = np.array(self.train_labels)    

        # load test data

        img, lab = load_batch('./traffic-signs-data/test.p')
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data,dtype=np.float32)
        self.test_labels = np.array(self.test_labels)   

        # load validation data

        img, lab = load_batch('./traffic-signs-data/valid.p')
        self.validation_data.extend(img)
        self.validation_labels.extend(lab)

        self.validation_data = np.array(self.validation_data,dtype=np.float32)
        self.validation_labels = np.array(self.validation_labels)  

class GTSRB(Dataset):
    def __init__(self,mode):
        self.data = torch.from_numpy(getattr(readGTSRB(),'{}_data'.format(mode))).float().permute(0,3,1,2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readGTSRB(),'{}_labels'.format(mode))).long()
        # print(self.target[0])
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
    
    def __len__(self):
        return len(self.data)

def load_gtsrb(args):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(GTSRB('train'),batch_size=args.batch_size, shuffle=True, **kwargs)
    # train_loader = torch.utils.data.DataLoader(GTSRB('train'),batch_size=1, shuffle=True, **kwargs)

    # mean,std, minimum, maximum = utils.get_mean_and_std(GTSRB('train'))
    # print(mean,std, minimum, maximum)

    # valid_loader = torch.utils.data.DataLoader(GTSRB('validation'),
    # batch_size=1, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(GTSRB('test'),
    batch_size=1, shuffle=False, **kwargs)

    # return train_loader, valid_loader, test_loader
    return train_loader, test_loader


class readKUL:
    def __init__(self):
        self.train_data = np.load('KUL/train_data.npy')
        self.train_labels = np.load('KUL/train_labels.npy')
        self.test_data = np.load('KUL/test_data.npy')
        self.test_labels = np.load('KUL/test_labels.npy')

class KUL(Dataset):
    def __init__(self,mode):
        self.data = torch.from_numpy(getattr(readKUL(),'{}_data'.format(mode))).float().permute(0,3,1,2)
        self.target = torch.from_numpy(getattr(readKUL(),'{}_labels'.format(mode))).long()
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]        
        return x, y
    
    def __len__(self):
        return len(self.data)    

def load_kul(args):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(KUL('train'),
    batch_size=32, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(KUL('test'),
    batch_size=1, shuffle=False, **kwargs)

    return train_loader, test_loader