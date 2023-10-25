import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import importlib
import datagen
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
import os
import csv
import torchvision
from torch.utils.data import DataLoader
from dataloader import notMNIST
from scipy.stats import entropy
import time
import foolbox
import uap
from adv_patch_utils import *
import torchvision.utils as vutils
from PGD import PGD,FGSM
import random
import torchattacks
from statistics import mean


def load_args():

    parser = argparse.ArgumentParser(description='HyperGAN')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--input_size', default=32, type=int, help='input images size')
    parser.add_argument('--num_classes', default=43, type=int, help='number of classes')
    parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
    parser.add_argument('--s_mean', default=0, type=int, help='S sample mean')
    parser.add_argument('--s_std', default=1, type=int, help='S sample standard deviation')
    parser.add_argument('--s', default=256, type=int, help='S sample dimension')
    parser.add_argument('--bias', action='store_true', help='Include HyperGAN bias')
    parser.add_argument('--batch_size', default=32, type=int, help='network batch size')
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='smallnet_small', type=str, help='target name')
    parser.add_argument('--beta', default=1, type=int, help='lagrangian strength')
    parser.add_argument('--pretrain_e', action='store_true')
    parser.add_argument('--exp', default='0', type=str)
    parser.add_argument('--resume', default='saved_models/gtsrb/smallnet_small-0.9557823634204275.pt', type=str, help='resume from path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (optimizer)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--dataset', default='gtsrb', type=str, help='mnist, cifar, cifar_hidden')
    parser.add_argument('--attack', default='fgsm', type=str, help='fgsm,cw')
    parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
    parser.add_argument('--patch_size', type=float, default=0.1, help='patch size. E.g. 0.05 ~= 5% of image ')
    parser.add_argument('--patch_shape', type=int, default=1, help='patch shape')
    parser.add_argument('--min_val', type=float, default=-0.5, help='min value for dataset')
    parser.add_argument('--max_val', type=float, default=0.5, help='max value for dataset')
    parser.add_argument('--patch_target', type=int, default=0, help='The target class for adv patch')
    parser.add_argument('--conf_target', type=float, default=0.9, help='Stop attack on image when target classifier reaches this value for target class')

    args = parser.parse_args()

    '''set random seed and device'''
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)        
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        args.device = 'cuda:0'
    else:
        torch.manual_seed(args.manualSeed)
        args.device = 'cpu'
        
    set_ngen(args)
    
    return args


def set_ngen(args):
    if args.dataset=='gtsrb':
        args.num_classes=43
        args.input_size=32
        args.in_channels=3
    if args.dataset=='kul':
        args.num_classes=62
        args.input_size=32
        args.in_channels=3
    if args.dataset=='mnist':
        args.num_classes=10
        args.input_size=28
        args.in_channels=1
    if args.dataset=='cifar':
        args.num_classes=10
        args.input_size=32
        args.in_channels=3
    if args.target in ['smallnet','smallnet_small','small']:
        args.ngen = 3
    elif args.target in ['lenet', 'lenet_small','mednet','mednet_small']: 
        args.ngen = 5
    elif args.target in ['smallnet_large','mednet_large']: 
        args.ngen = 7
    else:
        raise ValueError
    return


def unnormalize(x,args):
    if args.dataset == 'gtsrb':
        x += .5
    else:
        x *= .3081
        x += .1307
    return x


def normalize(x,args):
    if args.dataset == 'gtsrb':
        x-=.5
    else:
        x -= .1307
        x /= .3081
    return x


def imshow(img,args):
    img = unnormalize(img,args)     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()
    plt.savefig("mygraph.png")


def write_csv(filename, preds):
  with open(filename, 'a', newline='') as csvfile:
      writer = csv.writer(csvfile, delimiter=',',
                              quotechar='|', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(preds)


def Softmax(outputs):
    S = torch.nn.Softmax(dim = 1)
    return S(outputs)


def h_loss(pred):
    hloss = 0.
    for e in pred:
        value,counts = np.unique(e, return_counts=True)
        counts=counts/len(e)
        hloss += entropy(counts)
        # print(counts,entropy(counts))
    return hloss/len(pred)


def num_unique(list): 
  
    unique_list = [] 
      
    for x in list: 
        if x not in unique_list: 
            unique_list.append(x) 
    return len(unique_list)


def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum


def PGD_adversarial_training(args):

    epsilon = 0.05
    set_ngen(args)
    train_loader, test_loader = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    
    models = importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    model = hypergan.model

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6,momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    for epoch in range(200):
        total = 0.
        correct = 0.
        adv_correct = 0.
        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            total += labels.size(0)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            clean_loss = criterion(outputs, labels)

            attack = PGD(epsilon=epsilon,p=np.inf,stepsize=2.5*epsilon/100,numIters=100)
            adv_data = attack.attack(data=inputs,target=labels,model=model)

            adv_outputs = model(adv_data)
            _, adv_predicted = torch.max(adv_outputs.data, 1)
            adv_correct += (adv_predicted == labels).sum().item()
            adv_loss = criterion(adv_outputs, labels)

            loss = clean_loss + adv_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = correct/total
        train_adv_acc = adv_correct/total

        correct = 0.
        total = 0.
        with torch.no_grad():
            for i, (inputs,labels) in enumerate(test_loader):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_acc = correct/total
        print('Accuracy on testing images: {} and training images: {} and adv images: {}, best test acc: {}'.format(test_acc,train_acc,train_adv_acc,best_test_acc))

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(),'./saved_models/{}/gtsrb-{}-pgd-adv-trained.pth'.format(args.dataset,args.target))


def test_PGD_adv_train(args):
    kappa = 0
    set_ngen(args)
    _, test_loader = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models = importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    model = hypergan.model
    
    model.load_state_dict(torch.load('saved_models/{}/gtsrb-{}-pgd-adv-trained.pth'.format(args.dataset,args.target), map_location=args.device))
    
    attack = torchattacks.CW(model, c=1, kappa=kappa, steps=100, lr=0.01)
    # attack = torchattacks.FGSM(model,eps=0.1)

    correct = 0.
    total = 0.
    for i, (data, target) in enumerate(test_loader):
        if i >= 1000:
            break
        else:
            data = data.to(args.device)
            target = target.to(args.device)
            
            # adv = attack(data,target)
            adv = attack.attack(data=data,target=target,model=model)
            
            _, predicted = torch.max(model(adv).data, 1)
            correct += (predicted.data.item() == target.data.item())
            total += 1
    print(correct/total)


def train_ensemble(args):

    set_ngen(args)
    models = importlib.import_module('models.{}'.format(args.target))
    
    train_loader, test_loader = getattr(datagen, 'load_{}'.format(args.dataset))(args)

    criterion = nn.CrossEntropyLoss()

    for model_index in range(0,20):
        hypergan = models.HyperGAN(args)
        model = hypergan.model
        # model = models.MedNet(args).to(args.device)
        
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-6,momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

        best_test_acc = 0.0
        for epoch in range(200):
            total = 0.
            correct = 0.
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_acc = correct/total

            correct = 0.
            total = 0.
            with torch.no_grad():
                for i, (inputs,labels) in enumerate(test_loader):
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            test_acc = correct/total
            print('Accuracy on testing images: {} and training images {}, best test acc: {}'.format(test_acc,train_acc,best_test_acc))

            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(),'./saved_models/test_ensemble_models_mednet/{}_model_{}.pth'.format(args.dataset,model_index))


def test_outlier_baseline_nmist(args):
    
    ''' prepare model's architecture '''
    models = importlib.import_module('models.{}'.format(args.target))   

    '''load notMNIST dataset'''
    path = os.path.join(os.path.dirname(__file__), 'data_nm/Test')
    test_dataset = notMNIST(path)
    test_loader_nm = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ''' load MNIST dataset'''
    _, test_loader_m = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    
    ''' baseline ROC '''
    for ts in range(90,101):
        Ts = ts/100
        fpr = 0.
        tpr = 0.
        for _, data in enumerate(test_loader_m, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            outputs = model_list[1](test_x)
            prob,_ = torch.max(Softmax(outputs).data, 1)
            if prob < Ts:
                fpr +=1

        for _, data in enumerate(test_loader_nm, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            outputs = model_list[1](test_x)
            prob,_ = torch.max(Softmax(outputs).data, 1)
            if prob < Ts:
                tpr +=1

        write_csv('anomaly_detection_baseline.csv',[Ts,fpr/len(test_loader_m),tpr/len(test_loader_nm)])


def test_outlier_deepMTD_nmnist(args):
    
    ''' prepare model's architecture '''
    models = importlib.import_module('models.{}'.format(args.target))   

    '''load notMNIST dataset'''
    path = os.path.join(os.path.dirname(__file__), 'data_nm/Test')
    test_dataset = notMNIST(path)
    test_loader_nm = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ''' load MNIST dataset'''
    _, test_loader_m = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    
    '''load deepMTD ensemble'''
    model_list = []
    for i in range(20):
        hypergan_tmp = models.HyperGAN(args)
        classifier = hypergan_tmp.model
        classifier.load_state_dict(torch.load('saved_models/test_ensemble_models/mnist_model_'+str(i)+'.pth', map_location=args.device))
        model_list.append(classifier)

    '''distinct output'''
    outputs_list = []
    for _, data in enumerate(test_loader_nm, 0):
        test_x, test_y = data
        test_x=test_x.to(args.device)
        test_y=test_y.to(args.device)
        pred = []
        for i in range(len(model_list)):
            outputs = model_list[i](test_x)
            _, predicted = torch.max(outputs.data, 1)
            pred.append(predicted.data.item())
        # pred_list.append(pred)
        num_distinct_output = num_unique(pred)
        outputs_list.append(num_distinct_output)
    print(mean(outputs_list))
    #     write_csv('num_distinct_output_standard_ensemble.csv',[num_distinct_output])

    ''' plot ROC '''
    for ts in range(50,101,5):
        Ts = ts/100

        fpr_t_en = 0.
        tpr_t_en = 0.

        pred_list = []
        for _, data in enumerate(test_loader_m, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for i in range(len(model_list)):
                outputs = model_list[i](test_x)
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/len(model_list) < Ts:
                fpr_t_en +=1
        # print("h loss of traditional ensemble: ",h_loss(pred_list))

        pred_list = []
        for _, data in enumerate(test_loader_nm, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for i in range(len(model_list)):
                outputs = model_list[i](test_x)
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/len(model_list) < Ts:
                tpr_t_en +=1
        # print("h loss of traditional ensemble: ",h_loss(pred_list))

        # write_csv('anomaly_detection_tradition_ensemble.csv',[Ts,fpr_t_en/len(test_loader_m),tpr_t_en/len(test_loader_nm)])
        print(Ts,fpr_t_en/len(test_loader_m),tpr_t_en/len(test_loader_nm))
  
    
def test_outlier_hypernet_nmist(args):
# python3 test_ensemble.py --cuda --dataset=mnist --target=smallnet --resume='saved_models/mnist/10/small_standard-0.990159375.pt' --s_std=10  --batch_size=100

    ''' prepare model's architecture '''
    models = importlib.import_module('models.{}'.format(args.target))   

    '''load notMNIST dataset'''
    path = os.path.join(os.path.dirname(__file__), 'data_nm/Test')
    test_dataset = notMNIST(path)
    test_loader_nm = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ''' load MNIST dataset'''
    _, test_loader_m = getattr(datagen, 'load_{}'.format(args.dataset))(args)


    '''load HyperNet ensemble'''
    hypergan = models.HyperGAN(args)
    if args.resume is not None:
        hypergan.restore_models(args)
    generator = hypergan.generator
    mixer = hypergan.mixer

    s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
    codes = mixer(s)
    # codes = codes.view(args.batch_size, args.ngen, args.z)
    # codes = torch.stack([codes[:, i] for i in range(args.ngen)])
    params = generator(codes)

    '''distinct output'''
    outputs_list = []
    for _, data in enumerate(test_loader_nm, 0):

        s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
        codes = mixer(s)
        # codes = codes.view(args.batch_size, args.ngen, args.z)
        # codes = torch.stack([codes[:, i] for i in range(args.ngen)])
        params = generator(codes)

        test_x, _ = data
        test_x=test_x.to(args.device)
        pred = []
        for (layers) in zip(*params):
            outputs = hypergan.eval_f(args, layers, test_x)
            _, predicted = torch.max(outputs.data, 1)
            pred.append(predicted.data.item())
        num_distinct_output = num_unique(pred)
        # print(pred,num_distinct_output)
        write_csv('num_distinct_output_HyperGAN_ensemble.csv',[num_distinct_output])
        outputs_list.append(num_distinct_output)
    print(mean(outputs_list))


    ''' plot ROC '''
    for num_models in [20]:
        print('****** N = {} ******'.format(num_models))
        for ts in range(50,110,10):
            Ts = ts/100

            fpr_h_en = 0.
            tpr_h_en = 0.
            pred_list = []
            for _, data in enumerate(test_loader_m, 0):
                test_x, test_y = data
                test_x=test_x.to(args.device)
                test_y=test_y.to(args.device)
                pred = []
                pred_cnt = 0
                for (layers) in zip(*params):
                    if pred_cnt >= num_models:
                        break
                    else:
                        pred_cnt += 1
                        outputs = hypergan.eval_f(args, layers, test_x)
                        _, predicted = torch.max(outputs.data, 1)
                        pred.append(predicted.data.item())
                pred_list.append(pred)
                if find_majority(pred)[1]/num_models < Ts:
                    fpr_h_en+=1
            # print("h loss of HyperNet ensemble: ",h_loss(pred_list))
            
            pred_list = []
            for _, data in enumerate(test_loader_nm, 0):
                
                s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
                codes = mixer(s)
                # codes = codes.view(args.batch_size, args.ngen, args.z)
                # codes = torch.stack([codes[:, i] for i in range(args.ngen)])
                params = generator(codes)
                
                test_x, test_y = data
                test_x=test_x.to(args.device)
                test_y=test_y.to(args.device)
                pred = []
                pred_cnt = 0
                for (layers) in zip(*params):
                    if pred_cnt >= num_models:
                        break
                    else:
                        pred_cnt += 1
                        outputs = hypergan.eval_f(args, layers, test_x)
                        _, predicted = torch.max(outputs.data, 1)
                        pred.append(predicted.data.item())
                pred_list.append(pred)
                if find_majority(pred)[1]/num_models < Ts:
                    tpr_h_en+=1
            # print("h loss of HyperNet ensemble: ",h_loss(pred_list))
            # write_csv('anomaly_detection_hypernet_ensemble-N{}.csv'.format(args.batch_size),[Ts,fpr_h_en/len(test_loader_m),tpr_h_en/len(test_loader_nm)])
            print(Ts,fpr_h_en/len(test_loader_m),tpr_h_en/len(test_loader_nm))


def test_outlier_FADA_nmnist(args):
    
    classifier_list = []
    encoder_list = []
    for i in range(20):
        main_models = importlib.import_module('models.FADA_main_models')

        classifier=main_models.Classifier()
        encoder=main_models.Encoder()

        classifier.load('saved_models/FADA-few-shot-retraining/mnist/classifier_mnist_{}.pt'.format(i))
        encoder.load('saved_models/FADA-few-shot-retraining/mnist/encoder_mnist_{}.pt'.format(i))

        classifier.to(args.device)
        encoder.to(args.device)
        
        classifier_list.append(classifier)
        encoder_list.append(encoder)

    '''load notMNIST dataset'''
    path = os.path.join(os.path.dirname(__file__), 'data_nm/Test')
    test_dataset = notMNIST(path)
    test_loader_nm = DataLoader(test_dataset, batch_size=1, shuffle=True)

    ''' load MNIST dataset'''
    # _, test_loader_m = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    test_loader_m=datagen.mnist_dataloader(batch_size=1,train=False)

    acc=0
    for data,labels in test_loader_m:
        data=data.to(args.device)
        labels=labels.to(args.device)
        y_test_pred=classifier_list[1](encoder_list[1](data))
        acc+=(torch.max(y_test_pred,1)[1]==labels)

    print(acc)

    for ts in range(50,101,5):
        Ts = ts/100

        fpr_t_en = 0.
        tpr_t_en = 0.

        pred_list = []
        for _, data in enumerate(test_loader_m, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for i in range(len(classifier_list)):
                outputs = classifier_list[i](encoder_list[i](test_x))
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/len(classifier_list) < Ts:
                fpr_t_en +=1

        pred_list = []
        for _, data in enumerate(test_loader_nm, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for i in range(len(classifier_list)):
                outputs = classifier_list[i](encoder_list[i](test_x))
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/len(classifier_list) < Ts:
                tpr_t_en +=1

        write_csv('anomaly_detection_FADA_ensemble.csv',[Ts,fpr_t_en/len(test_loader_m),tpr_t_en/len(test_loader_nm)])

    outputs_list = []
    for _, data in enumerate(test_loader_nm, 0):
        test_x, test_y = data
        test_x=test_x.to(args.device)
        test_y=test_y.to(args.device)
        pred = []
        for i in range(len(classifier_list)):
            outputs = classifier_list[i](encoder_list[i](test_x))
            _, predicted = torch.max(outputs.data, 1)
            pred.append(predicted.data.item())
        # pred_list.append(pred)
        num_distinct_output = num_unique(pred)
        outputs_list.append(num_distinct_output)
    # print(mean(outputs_list))
        write_csv('num_distinct_output_FADA_ensemble.csv',[num_distinct_output])


def test_outlier_baseline_kul(args):

    ''' prepare model's architecture '''
    models = importlib.import_module('models.{}'.format(args.target))   

    ''' load KUL dataset'''
    _, test_loader_kul = getattr(datagen, 'load_{}'.format(args.dataset))(args)

    ''' load nKUL dataset'''
    _, test_loader_nkul = getattr(datagen, 'load_nkul')(args)   

    ''' baseline ROC '''
    for ts in range(90,101):
        Ts = ts/100
        fpr = 0.
        tpr = 0.
        for _, data in enumerate(test_loader_kul, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            outputs = model_list[1](test_x)
            prob,_ = torch.max(Softmax(outputs).data, 1)
            if prob < Ts:
                fpr +=1

        for _, data in enumerate(test_loader_nkul, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            outputs = model_list[1](test_x)
            prob,_ = torch.max(Softmax(outputs).data, 1)
            if prob < Ts:
                tpr +=1

        write_csv('kul_outlier_detection_baseline.csv',[Ts,fpr/len(test_loader_kul),tpr/len(test_loader_nkul)])


def test_outlier_deepMTD_kul(args):

    ''' prepare model's architecture '''
    models = importlib.import_module('models.{}'.format(args.target))   

    ''' load KUL dataset'''
    _, test_loader_kul = getattr(datagen, 'load_{}'.format(args.dataset))(args)

    ''' load nKUL dataset'''
    _, test_loader_nkul = getattr(datagen, 'load_nkul')(args)   


    '''load deepMTD ensemble'''
    model_list = []
    for i in range(5):
        hypergan_tmp = models.HyperGAN(args)
        classifier = hypergan_tmp.model
        classifier.load_state_dict(torch.load('saved_models/kul-single-mednet_'+str(i)+'.pth', map_location=args.device))
        model_list.append(classifier) 
        
    '''plot ROC'''
    for ts in range(50,101,5):
        Ts = ts/100

        fpr_t_en = 0.
        tpr_t_en = 0.

        pred_list = []
        for _, data in enumerate(test_loader_m, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for i in range(len(model_list)):
                outputs = model_list[i](test_x)
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/len(model_list) < Ts:
                fpr_t_en +=1
        print("h loss of traditional ensemble: ",h_loss(pred_list))

        pred_list = []
        for _, data in enumerate(test_loader_nm, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for i in range(len(model_list)):
                outputs = model_list[i](test_x)
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/len(model_list) < Ts:
                tpr_t_en +=1
        print("h loss of traditional ensemble: ",h_loss(pred_list))

        write_csv('anomaly_detection_tradition_ensemble.csv',[Ts,fpr_t_en/len(test_loader_m),tpr_t_en/len(test_loader_nm)])


def test_outlier_hypernet_kul(args):
    
    ''' prepare model's architecture '''
    models = importlib.import_module('models.{}'.format(args.target))   

    ''' load KUL dataset'''
    _, test_loader_kul = getattr(datagen, 'load_{}'.format(args.dataset))(args)

    ''' load nKUL dataset'''
    _, test_loader_nkul = getattr(datagen, 'load_nkul')(args)   

    '''load HyperNet ensemble'''
    hypergan = models.HyperGAN(args)
    if args.resume is not None:
        hypergan.restore_models(args)
    generator = hypergan.generator
    mixer = hypergan.mixer

    s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
    codes = mixer(s)
    codes = codes.view(args.batch_size, args.ngen, args.z)
    codes = torch.stack([codes[:, i] for i in range(args.ngen)])
    params = generator(codes)    
    
    for ts in range(50,101,5):
        Ts = ts/100

        fpr_h_en = 0.
        tpr_h_en = 0.
        pred_list = []
        for _, data in enumerate(test_loader_kul, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for (layers) in zip(*params):
                outputs = hypergan.eval_f(args, layers, test_x)
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/args.batch_size < Ts:
                fpr_h_en+=1
        # print("h loss of HyperNet ensemble: ",h_loss(pred_list))
        
        pred_list = []
        for _, data in enumerate(test_loader_nkul, 0):
            test_x, test_y = data
            test_x=test_x.to(args.device)
            test_y=test_y.to(args.device)
            pred = []
            for (layers) in zip(*params):
                outputs = hypergan.eval_f(args, layers, test_x)
                _, predicted = torch.max(outputs.data, 1)
                pred.append(predicted.data.item())
            pred_list.append(pred)
            if find_majority(pred)[1]/args.batch_size < Ts:
                tpr_h_en+=1
        # print("h loss of HyperNet ensemble: ",h_loss(pred_list))
        write_csv('kul-outlier_detection_hypernet_ensemble-N{}.csv'.format(args.batch_size),[Ts,fpr_h_en/len(test_loader_kul),tpr_h_en/len(test_loader_nkul)])


def test_ensemble_acc(args):
    
    _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    
    models = importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    generator = hypergan.generator
    mixer = hypergan.mixer

    if args.resume is not None:
        hypergan.restore_models(args)


    with torch.no_grad():
        
        s = torch.normal(mean=args.s_mean, std=args.s_std, size=(100, args.s)).to(args.device)
        codes = mixer(s)
        codes = codes.view(100, args.ngen, args.z)
        codes = torch.stack([codes[:, i] for i in range(args.ngen)])
        params = generator(codes)
        
        # for num_models in range(10,101,10):
        for num_models in [3]:
            test_acc = 0.
            for i, (data, target) in enumerate(testset):
                data = data.to(args.device)
                target = target.to(args.device)
                pred_list = []
                pred_cnt = 0
                for (layers) in zip(*params):
                    if pred_cnt >= num_models:
                        break
                    else:
                        pred_cnt += 1
                        out = hypergan.eval_f(args, layers, data)
                        pred_list.append(out.data.max(1)[1].item())
                        # print(pred,target.data.view_as(pred))
                if find_majority(pred_list)[0]==target.data.item():
                    test_acc += 1
            test_acc /= len(testset.dataset)
            # print('Test Accuracy using {} models: {}'.format(args.batch_size, test_acc))
            write_csv('acc_vs_N.csv',[num_models, test_acc])


def test_inference_time(args):

    set_ngen(args)
    models = importlib.import_module('models.{}'.format(args.target))


    hypergan = models.HyperGAN(args)
    generator = hypergan.generator
    mixer = hypergan.mixer

    if args.resume is not None:
        hypergan.restore_models(args)

    s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
    # start = time.time()
    for i in range(10):
        codes = mixer(s)
    # end = time.time()
    # print(end-start)

    start = time.time()
    codes = mixer(s)
    codes = codes.view(args.batch_size, args.ngen, args.z)
    codes = torch.stack([codes[:, i] for i in range(args.ngen)])
    # start = time.time()
    params = generator(codes)
    end = time.time()
    print(end-start)


def generate_PGD(args):
    
    epsilon = 0.05
    _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    
    models=importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    base_model = hypergan.model
    base_model.load_state_dict(torch.load('saved_models/test_ensemble_models/{}_model_0.pth'.format(args.dataset)))
    base_model = base_model.eval()

    attack = PGD(epsilon=epsilon,p=np.inf,stepsize=2.5*epsilon/100,numIters=100)
    # attack = FGSM(epsilon=epsilon)

    adv_x=[]
    adv_y=[]
    correct = 0
    for i, (data, target) in enumerate(testset):
        if i >= 100:
            break
        data = data.to(args.device)
        target = target.to(args.device)

        adv = attack.attack(data=data,target=target,model=base_model)

        _, predicted = torch.max(base_model(adv).data, 1)
        correct += (predicted.data.item() == target.data.item())
        
        adv_x.append(adv)
        adv_y.append(target)
    print(correct/len(adv_x))
    np.save('./{}-{}-x.npy'.format(args.dataset,args.attack),adv_x)
    np.save('./{}-{}-y.npy'.format(args.dataset,args.attack),adv_y)


def generate_adv(args):

    _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    
    '''set up base model'''
    models=importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    b_model = hypergan.model
    b_model.load_state_dict(torch.load('saved_models/test_ensemble_models/{}_model_0.pth'.format(args.dataset)))
    b_model = b_model.eval()
    base_model = foolbox.models.PyTorchModel(b_model, bounds=(0,1),device=args.device)

    '''set up attack'''
    # attack = foolbox.attacks.PGD()
    attack = foolbox.attacks.FGSM()
    # attack = foolbox.attacks.L2CarliniWagnerAttack()
    # attack = foolbox.attacks.LinfDeepFoolAttack()

    adv_x=[]
    adv_y=[]
    for i, (data, target) in enumerate(testset):
        if i >= 100:
            break    
        adv_batch_x = data.to(args.device)
        adv_batch_y = target.to(args.device)
        _,adv_batch_x,is_adv=attack(base_model,adv_batch_x, adv_batch_y,epsilons=0.1)
        print(is_adv)
        adv_x.append(adv_batch_x)
        adv_y.append(adv_batch_y)
    np.save('./{}-{}-x.npy'.format(args.dataset,args.attack),adv_x)
    np.save('./{}-{}-y.npy'.format(args.dataset,args.attack),adv_y)


def generate_adv_UAP(args):

    train_loader, test_loader = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models=importlib.import_module('models.{}'.format(args.target))
    
    '''set up base model'''
    hypergan = models.HyperGAN(args)
    b_model = hypergan.model
    
    # b_model.load_state_dict(torch.load('saved_models/test_ensemble_models_mednet/{}_model_0.pth'.format(args.dataset)))
    b_model.load_state_dict(torch.load('saved_models/{}/gtsrb-{}-pgd-adv-trained.pth'.format(args.dataset,args.target), map_location=args.device))
    # b_model.load_state_dict(torch.load('saved_models/test_ensemble_models/{}_model_0.pth'.format(args.dataset)))

    uap.generate(args,train_loader,test_loader,b_model,xi=0.05)

    # _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)

    # uap = torch.from_numpy(np.load('./{}-uap-{}-{}.npy'.format(args.dataset,'1000','0.1'))).to(args.device)

    # adv_x=[]
    # adv_y=[]
    # for i, (data, target) in enumerate(testset):
    #     if i >= 1000:
    #         break    
    #     # adv_batch_x = data.cpu().numpy()
    #     # adv_batch_y = target.cpu().numpy()
    #     adv_batch_x = data.to(args.device)
    #     adv_batch_y = target.to(args.device)
    #     adv_batch_x += uap
    #     adv_x.append(adv_batch_x)
    #     adv_y.append(adv_batch_y)
    # np.save('./{}-uap-{}-{}-x.npy'.format(args.dataset,'1000','0.1'),adv_x)
    # np.save('./{}-uap-y.npy'.format(args.dataset),adv_y)


def generate_adv_patch_examples(args):

    adv_xs=[]
    adv_ys=[]

    _, test_loader = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models=importlib.import_module('models.{}'.format(args.target))

    hypergan = models.HyperGAN(args)
    netClassifier = hypergan.model
    netClassifier.load_state_dict(torch.load('saved_models/test_ensemble_models/{}_model_0.pth'.format(args.dataset)))
    netClassifier.eval()

    num_img_tst = 1000

    if args.patch_type == 'circle':
        _, args.patch_shape = init_patch_circle(args.input_size, args.patch_size) 
    elif args.patch_type == 'square':
        _, args.patch_shape = init_patch_square(args.input_size, args.patch_size)

    patch = np.load('adv_patch_0.3313343328335832.npy')

    test_success = 0
    test_total = 0
    for batch_idx, (data, labels) in enumerate(test_loader):
        if test_total >num_img_tst:
            break
        elif labels.data[0] != args.patch_target:
            data = data.to(args.device)
            labels = labels.to(args.device)
            data, labels = Variable(data), Variable(labels)
            prediction = netClassifier(data)
            if prediction.data.max(1)[1][0] != labels.data[0]:
                continue
            test_total += 1 

            data_shape = data.data.cpu().numpy().shape
            if args.patch_type == 'circle':
                patch, mask, args.patch_shape = circle_transform(patch, data_shape, args.patch_shape, args.input_size)
            elif args.patch_type == 'square':
                patch, mask = square_transform(patch, data_shape, args.patch_shape, args.input_size)
            patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
            patch, mask = patch.to(args.device), mask.to(args.device)
            patch, mask = Variable(patch), Variable(mask)

            adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
            adv_x = torch.clamp(adv_x, args.min_val, args.max_val)

            adv_xs.append(adv_x)
            adv_ys.append(labels.data[0])

            adv_label = netClassifier(adv_x).data.max(1)[1][0]
            
            # if adv_label == args.patch_target:
            #     test_success += 1
            if adv_label != labels.data[0]:
                test_success += 1

            masked_patch = torch.mul(mask, patch)
            patch = masked_patch.data.cpu().numpy()
            new_patch = np.zeros(args.patch_shape)
            for i in range(new_patch.shape[0]): 
                for j in range(new_patch.shape[1]): 
                    new_patch[i][j] = submatrix(patch[i][j])
    
            patch = new_patch

    print("Test Success: {:.3f}".format(test_success/test_total))
    np.save('./{}-patch-{}-{}-x.npy'.format(args.dataset,'2000',args.patch_size),adv_xs)
    np.save('./{}-patch-y.npy'.format(args.dataset),adv_ys)


def generate_adv_patch(args):
    print(args)

    num_img_trn = 2000
    num_img_tst = 2000

    train_loader, test_loader = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models=importlib.import_module('models.{}'.format(args.target))

    hypergan = models.HyperGAN(args)
    netClassifier = hypergan.model
    netClassifier.load_state_dict(torch.load('saved_models/test_ensemble_models/{}_model_0.pth'.format(args.dataset)))
    # netClassifier.load_state_dict(torch.load('saved_models/test_ensemble_models_mednet/{}_model_0.pth'.format(args.dataset)))
    # netClassifier.load_state_dict(torch.load('saved_models/{}/gtsrb-{}-pgd-adv-trained.pth'.format(args.dataset,args.target), map_location=args.device))
    netClassifier.eval()

    if args.patch_type == 'circle':
        patch, args.patch_shape = init_patch_circle(args.input_size, args.patch_size) 
    elif args.patch_type == 'square':
        patch, args.patch_shape = init_patch_square(args.input_size, args.patch_size) 

    best_test_acc = 0.

    for epoch in range(args.epochs):
        success = 0
        total = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            if total >num_img_trn:
                break
            elif labels.data[0] != args.patch_target:
                data = data.to(args.device)
                labels = labels.to(args.device)
                data, labels = Variable(data), Variable(labels)
                prediction = netClassifier(data)

                # only compute adversarial examples on examples that are originally classified correctly
                if prediction.data.max(1)[1][0] != labels.data[0]:
                    continue
                total += 1

                # transform path
                data_shape = data.data.cpu().numpy().shape
                if args.patch_type == 'circle':
                    patch, mask, args.patch_shape = circle_transform(patch, data_shape, args.patch_shape, args.input_size)
                elif args.patch_type == 'square':
                    patch, mask  = square_transform(patch, data_shape, args.patch_shape, args.input_size)
                patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
                patch, mask = patch.to(args.device), mask.to(args.device)
                patch, mask = Variable(patch), Variable(mask)
                adv_x, mask, patch = patch_attack(data,labels, patch, mask, netClassifier,args)
                
                adv_label = netClassifier(adv_x).data.max(1)[1][0]
                ori_label = labels.data[0]
                
                # if adv_label == args.patch_target:
                #     success += 1
                if adv_label != labels.data[0]:
                    success += 1
                
                # if labels.data[0].item()==1:
                if labels.data[0].item()==12:
                    print(adv_label)
                    vutils.save_image(data.data, "original.png", normalize=True)
                    vutils.save_image(adv_x.data, "adversarial.png", normalize=True)
                    
                
                masked_patch = torch.mul(mask, patch)
                patch = masked_patch.data.cpu().numpy()
                new_patch = np.zeros(args.patch_shape)
                for i in range(new_patch.shape[0]): 
                    for j in range(new_patch.shape[1]): 
                        new_patch[i][j] = submatrix(patch[i][j])
        
                patch = new_patch

        print("Train Patch Success: {:.3f}".format(success/total))

        test_success = 0
        test_total = 0
        for batch_idx, (data, labels) in enumerate(test_loader):
            if test_total >num_img_tst:
                break
            elif labels.data[0] != args.patch_target:
                data = data.to(args.device)
                labels = labels.to(args.device)
                data, labels = Variable(data), Variable(labels)
                prediction = netClassifier(data)
                if prediction.data.max(1)[1][0] != labels.data[0]:
                    continue
                test_total += 1 

                data_shape = data.data.cpu().numpy().shape
                if args.patch_type == 'circle':
                    patch, mask, args.patch_shape = circle_transform(patch, data_shape, args.patch_shape, args.input_size)
                elif args.patch_type == 'square':
                    patch, mask = square_transform(patch, data_shape, args.patch_shape, args.input_size)
                patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
                patch, mask = patch.to(args.device), mask.to(args.device)
                patch, mask = Variable(patch), Variable(mask)

                adv_x = torch.mul((1-mask),data) + torch.mul(mask,patch)
                adv_x = torch.clamp(adv_x, args.min_val, args.max_val)

                adv_label = netClassifier(adv_x).data.max(1)[1][0]
                ori_label = labels.data[0]
                
                # if adv_label == args.patch_target:
                #     test_success += 1
                if adv_label != labels.data[0]:
                    test_success += 1

                masked_patch = torch.mul(mask, patch)
                patch = masked_patch.data.cpu().numpy()
                new_patch = np.zeros(args.patch_shape)
                for i in range(new_patch.shape[0]): 
                    for j in range(new_patch.shape[1]): 
                        new_patch[i][j] = submatrix(patch[i][j])
        
                patch = new_patch

        print("Test Success: {:.3f}".format(test_success/test_total))
        if test_success/test_total > best_test_acc:
            best_test_acc = test_success/test_total
            np.save('adv_patch_{}.npy'.format(best_test_acc),patch)


def test_adv_single_model(args):

    # _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models=importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    b_model = hypergan.model
    b_model.load_state_dict(torch.load('saved_models/test_ensemble_models/{}_model_1.pth'.format(args.dataset)))

    adv_x = np.load('./{}-{}-x.npy'.format(args.dataset,args.attack),allow_pickle=True)
    adv_y = np.load('./{}-{}-y.npy'.format(args.dataset,args.attack),allow_pickle=True)
    # adv_x = adv_x[:,1]
    
    correct = 0
    for adv_index in range(len(adv_x)):
        adv_batch_x = torch.Tensor(adv_x[adv_index]).to(args.device)
        # adv_batch_x = adv_x[adv_index].to(args.device)
        adv_batch_y = torch.Tensor(adv_y[adv_index]).to(args.device)

        out = b_model(adv_batch_x)
        _, pred = torch.max(out.data, 1)
        if pred.data.item() == adv_y[adv_index][0]:
            correct += 1
    print(correct,len(adv_x))


def test_adv_hypernet(args):

    _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models=importlib.import_module('models.{}'.format(args.target))

    hypergan = models.HyperGAN(args)
    mixer = hypergan.mixer
    generator = hypergan.generator
    if args.resume is not None:
        hypergan.restore_models(args)

    for num_models in range(10,210,10):
        args.batch_size = num_models
        s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
        codes = mixer(s)
        # codes = codes.view(args.batch_size, args.ngen, args.z)
        # codes = torch.stack([codes[:, i] for i in range(args.ngen)])
        params = generator(codes)

        for ts in range(50,110,10):
        # for ts in range(55,100,10):
            Ts = ts / 100
            '''Test ensemble on adv example'''
            true_positive=0
            true_positive_s=0
            true_positive_f=0
            false_negative=0
            false_negative_s=0
            false_negative_f=0

            # adv_x = np.load('./{}-{}-x.npy'.format(args.dataset,args.attack),allow_pickle=True)
            # adv_y = np.load('./{}-{}-y.npy'.format(args.dataset,args.attack),allow_pickle=True)
            # adv_x = np.load('./gtsrb-patch-2000-0.05-x.npy',allow_pickle=True)
            # adv_y = np.load('./gtsrb-patch-y.npy',allow_pickle=True)
            # adv_x = np.load('./gtsrb-uap-1000-0.1-x.npy',allow_pickle=True)
            # adv_y = np.load('./gtsrb-uap-y.npy',allow_pickle=True)

            adv_x = np.load('l2_ensemble_attack_gtsrb_k_0_x.npy')
            adv_x = np.expand_dims(adv_x,axis=1)
            adv_y = np.load('l2_ensemble_attack_gtsrb_k_0_y.npy')
            adv_y = np.expand_dims(np.argmax(adv_y,axis=1),axis=1)

            for adv_index in range(len(adv_x)):
                # print("!!",base_model.forward(np.array(adv_x[adv_index])).argmax(axis=-1)[0],adv_y[adv_index])
                adv_batch_x = torch.Tensor(adv_x[adv_index]).to(args.device)
                adv_batch_y = torch.Tensor(adv_y[adv_index]).to(args.device)
                # adv_batch_x = adv_x[adv_index].to(args.device)
                # adv_batch_y = adv_y[adv_index].to(args.device)

                pred_list_hyper=[]    
                for (layers) in zip(*params):
                    out = hypergan.eval_f(args, layers, adv_batch_x)
                    pred = out.data.max(1, keepdim=True)[1]
                    pred_list_hyper.append(pred.data.item())
                # print(b_model(adv_batch_x).argmax(axis=-1).data.item(),pred_list_hyper,adv_batch_y.data.item())
                # print(pred_list_hyper,adv_batch_y.data.item())
                
                if find_majority(pred_list_hyper)[1]/args.batch_size < Ts:
                    true_positive+=1
                    if find_majority(pred_list_hyper)[0]==adv_batch_y.data.item():
                        true_positive_s+=1
                    else:
                        true_positive_f+=1
                else:
                    false_negative+=1
                    if find_majority(pred_list_hyper)[0]==adv_batch_y.data.item():
                        false_negative_s+=1
                    else:
                        false_negative_f+=1

            true_positive_rate = true_positive/len(adv_x)
            if true_positive != 0:
                true_positive_correct_rate = true_positive_s/true_positive
                true_positive_wrong_rate = true_positive_f/true_positive
            else:
                true_positive_correct_rate = 0
                true_positive_wrong_rate = 0

            false_negative_rate = false_negative/len(adv_x)
            if false_negative != 0:
                false_negative_correct_rate = false_negative_s/false_negative
                false_negative_wrong_rate = false_negative_f/false_negative
            else:
                false_negative_correct_rate = 0
                false_negative_wrong_rate = 0

            print(num_models,Ts,true_positive_rate,true_positive_correct_rate,true_positive_wrong_rate,false_negative_rate,false_negative_correct_rate,false_negative_wrong_rate)
 
            
def test_adv_deepMTD(args):

    # python3 test_ensemble.py --cuda --dataset=gtsrb --target=smallnet_small --resume='./saved_models/gtsrb/smallnet_small-0.955096991290578.pt' --attack=cw
    # python3 test_ensemble.py --cuda --dataset=gtsrb --target=smallnet --resume='./saved_models/gtsrb/smallnet-0.954139449722882.pt' --attack=cw
    # python3 test_ensemble.py --cuda --dataset=mnist --target=smallnet --resume='./saved_models/mnist/smallnet-0.992775.pt' --attack=cw
    # python3 test_ensemble.py --cuda --dataset=mnist --target=smallnet_small --resume='./saved_models/mnist/smallnet_small-0.992528125.pt' --attack=cw

    _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    models=importlib.import_module('models.{}'.format(args.target))

    '''prepare tradition ensemble'''
    model_list = []
    for i in range(20):
        hypergan_tmp = models.HyperGAN(args)
        classifier = hypergan_tmp.model
        # classifier.load_state_dict(torch.load('saved_models/test_ensemble_models_mednet/gtsrb_model_'+str(i)+'.pth', map_location=args.device))
        classifier.load_state_dict(torch.load('saved_models/test_ensemble_models/gtsrb_model_'+str(i)+'.pth', map_location=args.device))
        model_list.append(classifier)

    for ts in range(50,110,10):
    # for ts in range(51,60):
        Ts = ts / 100
        '''Test ensemble on adv example'''
        true_positive=0
        true_positive_s=0
        true_positive_f=0
        false_negative=0
        false_negative_s=0
        false_negative_f=0

        # adv_x = np.load('./{}-{}-x.npy'.format(args.dataset,args.attack),allow_pickle=True)
        # adv_y = np.load('./{}-{}-y.npy'.format(args.dataset,args.attack),allow_pickle=True)
        # adv_x = np.load('./gtsrb-uap-1000-0.1-x.npy',allow_pickle=True)
        # adv_y = np.load('./gtsrb-uap-y.npy',allow_pickle=True)
        adv_x = np.load('./gtsrb-patch-2000-0.05-x.npy',allow_pickle=True)
        adv_y = np.load('./gtsrb-patch-y.npy',allow_pickle=True)

        # adv_x = np.load('l2_ensemble_attack_gtsrb_k_0_x.npy')
        # adv_x = np.expand_dims(adv_x,axis=1)
        # adv_y = np.load('l2_ensemble_attack_gtsrb_k_0_y.npy')
        # adv_y = np.expand_dims(np.argmax(adv_y,axis=1),axis=1)

        for adv_index in range(len(adv_x)):
            # adv_batch_x = torch.Tensor(adv_x[adv_index]).to(args.device)
            # adv_batch_y = torch.Tensor(adv_y[adv_index]).to(args.device)
            adv_batch_x = adv_x[adv_index].to(args.device)
            adv_batch_y = adv_y[adv_index].to(args.device)

            pred_list_hyper=[]    
            for i in range(len(model_list)):
                out = model_list[i](adv_batch_x)
                _, pred = torch.max(out.data, 1)
                pred_list_hyper.append(pred.data.item())
            
            if find_majority(pred_list_hyper)[1]/len(model_list) < Ts:
                true_positive+=1
                if find_majority(pred_list_hyper)[0]==adv_batch_y.data.item():
                    true_positive_s+=1
                else:
                    true_positive_f+=1
            else:
                false_negative+=1
                if find_majority(pred_list_hyper)[0]==adv_batch_y.data.item():
                    false_negative_s+=1
                else:
                    false_negative_f+=1

        true_positive_rate = true_positive/len(adv_x)
        if true_positive != 0:
            true_positive_correct_rate = true_positive_s/true_positive
            true_positive_wrong_rate = true_positive_f/true_positive
        else:
            true_positive_correct_rate = 0
            true_positive_wrong_rate = 0

        false_negative_rate = false_negative/len(adv_x)
        if false_negative != 0:
            false_negative_correct_rate = false_negative_s/false_negative
            false_negative_wrong_rate = false_negative_f/false_negative
        else:
            false_negative_correct_rate = 0
            false_negative_wrong_rate = 0


        print(Ts,true_positive_rate,true_positive_correct_rate,true_positive_wrong_rate,false_negative_rate,false_negative_correct_rate,false_negative_wrong_rate)


def test_adv_FADA(args):

    ''' prepare model's architecture '''
    classifier_list = []
    encoder_list = []
    for i in range(20):
        main_models = importlib.import_module('models.FADA_main_models')

        classifier=main_models.Classifier_GTSRB()
        encoder=main_models.Encoder_GTSRB()

        classifier.load('saved_models/FADA-few-shot-retraining/gtsrb/classifier_gtsrb_{}.pt'.format(i))
        encoder.load('saved_models/FADA-few-shot-retraining/gtsrb/encoder_gtsrb_{}.pt'.format(i))

        classifier.to(args.device)
        encoder.to(args.device)
        
        classifier_list.append(classifier)
        encoder_list.append(encoder)



    _, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    # adv_x = np.load('./{}-{}-x.npy'.format(args.dataset,args.attack),allow_pickle=True)
    # adv_y = np.load('./{}-{}-y.npy'.format(args.dataset,args.attack),allow_pickle=True)

    # adv_x = np.load('./gtsrb-uap-1000-0.1-x.npy',allow_pickle=True)
    # adv_y = np.load('./gtsrb-uap-y.npy',allow_pickle=True)

    adv_x = np.load('./gtsrb-patch-2000-0.05-x.npy',allow_pickle=True)
    adv_y = np.load('./gtsrb-patch-y.npy',allow_pickle=True)

    for ts in range(50,110,10):
        Ts = ts / 100
        '''Test ensemble on adv example'''
        true_positive=0
        true_positive_s=0
        true_positive_f=0
        false_negative=0
        false_negative_s=0
        false_negative_f=0

        for adv_index in range(len(adv_x)):
            # adv_batch_x = torch.Tensor(adv_x[adv_index]).to(args.device)
            # adv_batch_y = torch.Tensor(adv_y[adv_index]).to(args.device)
            adv_batch_x = adv_x[adv_index].to(args.device)
            adv_batch_y = adv_y[adv_index].to(args.device)
            pred_list_hyper=[]    
            for i in range(len(classifier_list)):
                out = classifier_list[i](encoder_list[i](adv_batch_x))
                _, pred = torch.max(out.data, 1)
                pred_list_hyper.append(pred.data.item())
            
            if find_majority(pred_list_hyper)[1]/args.batch_size < Ts:
                true_positive+=1
                if find_majority(pred_list_hyper)[0]==adv_batch_y.data.item():
                    true_positive_s+=1
                else:
                    true_positive_f+=1
            else:
                false_negative+=1
                if find_majority(pred_list_hyper)[0]==adv_batch_y.data.item():
                    false_negative_s+=1
                else:
                    false_negative_f+=1

        true_positive_rate = true_positive/len(adv_x)
        if true_positive != 0:
            true_positive_correct_rate = true_positive_s/true_positive
            true_positive_wrong_rate = true_positive_f/true_positive
        else:
            true_positive_correct_rate = 0
            true_positive_wrong_rate = 0

        false_negative_rate = false_negative/len(adv_x)
        if false_negative != 0:
            false_negative_correct_rate = false_negative_s/false_negative
            false_negative_wrong_rate = false_negative_f/false_negative
        else:
            false_negative_correct_rate = 0
            false_negative_wrong_rate = 0

        print(Ts,true_positive_rate,true_positive_correct_rate,true_positive_wrong_rate,false_negative_rate,false_negative_correct_rate,false_negative_wrong_rate)



if __name__ == '__main__':
    args = load_args()
    train_ensemble(args)
