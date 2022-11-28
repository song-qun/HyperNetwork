import matplotlib
matplotlib.use('agg')
import torch
import argparse
import numpy as np
import importlib
import datagen
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import attack_model
import utils
import random

def load_args():

    parser = argparse.ArgumentParser(description='HyperNet')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--input_size', default=32, type=int, help='input images size')
    parser.add_argument('--num_classes', default=43, type=int, help='number of classes')
    parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
    parser.add_argument('--s_mean', default=0, type=int, help='S sample mean')
    parser.add_argument('--s_std', default=1, type=int, help='S sample standard deviation')
    parser.add_argument('--s', default=256, type=int, help='S sample dimension')
    parser.add_argument('--bias', action='store_true', help='Include HyperNet bias')
    parser.add_argument('--batch_size', default=32, type=int, help='network batch size')
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--target', default='smallnet_small', type=str, help='target name')
    parser.add_argument('--resume', default=None, type=str, help='resume from path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (optimizer)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--dataset', default='gtsrb', type=str, help='mnist, cifar, cifar_hidden')
    parser.add_argument('--diversity_lambda',type=int, default=1, help='diversity lambda')
    parser.add_argument('--netAttacker', default='', help="path to netAttacker (to continue training)")
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector for attack model')
    parser.add_argument('--attack_lr', type=float, default=0.0002, help='attack model learning rate, default=0.0002')
    parser.add_argument('--attack_beta1', type=float, default=0.5, help='attack model beta1 for adam. default=0.5')
    parser.add_argument('--attack_l2reg', type=float, default=0.01, help='attack model weight factor for l2 regularization')
    parser.add_argument('--min_val', type=float, default=-0.5, help='min value for dataset')
    parser.add_argument('--max_val', type=float, default=0.5, help='max value for dataset')
    parser.add_argument('--norm', type=str, default='linf', help='l2 or linf')
    parser.add_argument('--ldist_weight', type=float, default=4.0, help='how much to weight the ldist loss term')
    parser.add_argument('--adv_intensity', type=float, default=0.01, help='adversarial perturbation intensity')

    args = parser.parse_args()
    set_ngen(args)
    
    return args


def set_ngen(args):
    if args.dataset=='gtsrb':
        args.in_channels=3
        args.input_size=32
        args.num_classes=43
    if args.dataset=='kul':
        args.in_channels=3
        args.input_size=32
        args.num_classes=62
    if args.dataset=='mnist':
        args.in_channels=1
        args.input_size=28
        args.num_classes=10
    if args.target in ['smallnet','smallnet_small']:
        args.ngen = 3
    elif args.target in ['lenet', 'lenet_small','mednet','mednet_small']: 
        args.ngen = 5
    elif args.target in ['smallnet_large','mednet_large']: 
        args.ngen = 7
    else:
        raise ValueError
    return


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


def train(args):

    '''set random seed and device'''
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)        
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        args.device = 'cuda:2'
    else:
        torch.manual_seed(args.manualSeed)
        args.device = 'cpu'


    '''instantiate attack model and HyperNet'''
    netAttacker = attack_model._netAttacker(args.input_size,args.in_channels)
    if args.netAttacker != '':
        netAttacker.load_state_dict(torch.load(args.netAttacker))
    netAttacker = netAttacker.to(args.device)

    models = importlib.import_module('models.{}'.format(args.target))
    hypergan = models.HyperGAN(args)
    generator = hypergan.generator
    mixer = hypergan.mixer
    if args.resume is not None:
        hypergan.restore_models(args)


    """ attach optimizers """
    optimizerAttacker = torch.optim.Adam(netAttacker.parameters(), lr=args.attack_lr, betas=(args.attack_beta1, 0.999), weight_decay=args.attack_l2reg)
    
    optimQ = torch.optim.Adam(mixer.parameters(), lr=args.lr, weight_decay=args.wd)
    optimW = []
    for m in range(args.ngen):
        s = getattr(generator, 'W{}'.format(m+1))
        optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))
    
    schedulers = []
    steps = [10*i for i in range(1, 100)]
    for op in [optimQ, *optimW]:
        schedulers.append(utils.CyclicCosAnnealingLR(op, steps, eta_min=1e-8))

    best_test_acc, best_test_loss, = 0., np.inf
    best_test_adv_acc, best_test_adv_loss, = 0., np.inf
    

    '''prepare data'''
    trainset, testset = getattr(datagen, 'load_{}'.format(args.dataset))(args)
    noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1)
    noise = noise.to(args.device)
    noise = Variable(noise)
      

    print ('==> Begin Training')
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.epochs):
            acc = 0.
            adv_acc = 0.
            for batch_idx, (data, target) in enumerate(trainset):

                args.adv_intensity = random.choice([intensity / 100 for intensity in range(1,11,1)])

                data = data.to(args.device)
                target = target.to(args.device)

                s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
                codes = mixer(s)
                codes = codes.view(args.batch_size, args.ngen, args.z)
                codes = torch.stack([codes[:, i] for i in range(args.ngen)])
                params = generator(codes)

                '''diversity loss '''
                var = torch.var(torch.flatten(params[0],start_dim=1,end_dim=-1))+torch.var(torch.flatten(params[2],start_dim=1,end_dim=-1))+torch.var(torch.flatten(params[4],start_dim=1,end_dim=-1))
                varloss = args.diversity_lambda*torch.exp(-var)
                
                '''accuracy loss'''
                clf_loss = 0.
                for (layers) in zip(*params):
                    out = hypergan.eval_f(args, layers, data)
                    loss = F.cross_entropy(out, target)
                    pred = out.data.max(1, keepdim=True)[1]
                    acc += pred.eq(target.data.view_as(pred)).long().cpu().sum()
                    clf_loss += loss

                '''adversarial loss'''
                netAttacker.eval()
                noise.resize_(data.size(0), args.nz, 1, 1).normal_(0, 0.5)
                delta = netAttacker(noise).to(args.device)
                delta = delta*args.adv_intensity
                adv_sample_ = delta + data
                adv_sample = torch.clamp(adv_sample_, args.min_val, args.max_val)

                adv_loss = 0.
                for (layers) in zip(*params):
                    adv_out = hypergan.eval_f(args, layers, adv_sample)
                    loss = F.cross_entropy(adv_out, target)
                    adv_pred = adv_out.data.max(1, keepdim=True)[1]
                    adv_acc += adv_pred.eq(target.data.view_as(adv_pred)).long().cpu().sum()
                    adv_loss += loss

                """ calculate total loss on Q and G """
                Q_loss = varloss
                G_loss = clf_loss / args.batch_size
                A_loss = adv_loss / args.batch_size
                QGA_loss = Q_loss  + G_loss + A_loss
                QGA_loss.backward(retain_graph=True)
                
                optimQ.step()
                for optim in optimW:
                    optim.step()

                optimQ.zero_grad()
                for optim in optimW:
                    optim.zero_grad()
            
                '''update attack model to min adv loss'''
                netAttacker.train()
                # data = data.clone()
                # target = target.clone()
                params = [p.detach() for p in params]
                
                attack_loss = 0.
                for (layers) in zip(*params):
                    noise.resize_(data.size(0), args.nz, 1, 1).normal_(0, 0.5)
                    delta = netAttacker(noise).to(args.device)
                    delta = delta*args.adv_intensity
                    adv_sample_ = delta + data
                    adv_sample = torch.clamp(adv_sample_, args.min_val, args.max_val)                    
                    
                    adv_out = hypergan.eval_f(args, layers, adv_sample)
                    adv_pred = adv_out.data.max(1, keepdim=True)[1]
                    clean_out = hypergan.eval_f(args, layers, data)
                    clean_pred = clean_out.data.max(1, keepdim=True)[1]

                    no_idx = np.array(np.where(adv_pred.eq(clean_pred).cpu().numpy() == 1))[0].astype(int)

                    if len(no_idx)!=0:
                        # select the non adv examples to optimise on 
                        no_idx = torch.LongTensor(no_idx)
                        no_idx = no_idx.to(args.device)
                        no_idx = Variable(no_idx)
                        
                        data_tmp = torch.index_select(data, 0, no_idx)
                        target_tmp = torch.index_select(target, 0, no_idx)
                        adv_out_tmp = torch.index_select(adv_out, 0, no_idx)
                        delta_tmp = torch.index_select(delta, 0, no_idx)
                        adv_sample_tmp = torch.index_select(adv_sample, 0, no_idx)

                        adv_out_softmax = F.softmax(adv_out_tmp)
                        adv_out_np = adv_out_softmax.data.cpu().numpy()
                        curr_adv_label = Variable(torch.LongTensor(np.array([arr.argsort()[-1] for arr in adv_out_np]))).to(args.device)
                        targ_adv_label = Variable(torch.LongTensor(np.array([arr.argsort()[-2] for arr in adv_out_np]))).to(args.device)
                        # print(curr_adv_label,targ_adv_label,target_tmp)
                        
                        for sfmax_idx in range(len(target_tmp)):
                            if curr_adv_label[sfmax_idx] != target_tmp[sfmax_idx]:
                                targ_adv_label[sfmax_idx] = curr_adv_label[sfmax_idx]
                                curr_adv_label[sfmax_idx] = target_tmp[sfmax_idx]

                        # print(curr_adv_label,targ_adv_label,target_tmp)

                        curr_adv_pred = adv_out_softmax.gather(1, curr_adv_label.unsqueeze(1))
                        targ_adv_pred = adv_out_softmax.gather(1, targ_adv_label.unsqueeze(1))
                        
                        classifier_loss = torch.mean(curr_adv_pred-targ_adv_pred)
                
                        if args.norm == 'linf':
                            ldist_loss = args.ldist_weight*torch.max(torch.abs(adv_sample - data))
                        elif args.norm == 'l2':
                            ldist_loss = args.ldist_weight*torch.mean(torch.sqrt(torch.sum((adv_sample - data)**2)))
                        else:
                            print("Please define a norm (l2 or linf)")
                            exit()
                        
                        loss = classifier_loss + ldist_loss 
                        loss.backward(retain_graph=True)
                        optimizerAttacker.step()
                        optimizerAttacker.zero_grad()   
                        # c_loss.append(classifier_loss.data)    
                        attack_loss += loss        

                Att_loss = attack_loss / args.batch_size                               

            for scheduler in schedulers:
                scheduler.step()  

            acc /= len(trainset.dataset)*args.batch_size
            adv_acc /= len(trainset.dataset)*args.batch_size
        
            """ print training accuracy """
            print ('**************************************')
            print ('Epoch: {}'.format(epoch))
            print ('Train Acc: {}, Adv Acc: {}, G Loss: {}, Q Loss: {}, A Loss: {}, Attack Loss: {}'.format(acc, adv_acc, G_loss, Q_loss, A_loss, Att_loss))
            print ('best test loss: {}'.format(best_test_loss))
            print ('best test acc: {}'.format(best_test_acc))
            print ('best test adv loss: {}'.format(best_test_adv_loss))
            print ('best test adv acc: {}'.format(best_test_adv_acc))
            print ('**************************************')

            """ test random draw on testing set """

            

            netAttacker.eval()

            test_acc = 0.
            test_loss = 0.
            test_adv_acc = 0.
            test_adv_loss = 0.

            with torch.no_grad():
                for i, (data, target) in enumerate(testset):

                    data = data.to(args.device)
                    target = target.to(args.device)

                    s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
                    codes = mixer(s)
                    codes = codes.view(args.batch_size, args.ngen, args.z)
                    codes = torch.stack([codes[:, i] for i in range(args.ngen)])
                    params = generator(codes)

                    noise.resize_(data.size(0), args.nz, 1, 1).normal_(0, 0.5)
                    delta = netAttacker(noise).to(args.device)
                    delta = delta*args.adv_intensity
                    adv_sample_ = delta + data
                    adv_sample = torch.clamp(adv_sample_, args.min_val, args.max_val)

                    pred_list=[]
                    adv_pred_list=[]
                    for (layers) in zip(*params):
                        out = hypergan.eval_f(args, layers, data)
                        adv_out = hypergan.eval_f(args, layers, adv_sample)

                        loss = F.cross_entropy(out, target)
                        loss2 = F.cross_entropy(adv_out, target)

                        test_loss += loss.item()
                        test_adv_loss += loss2.item()

                        pred = out.data.max(1, keepdim=True)[1]
                        pred2 = adv_out.data.max(1, keepdim=True)[1]

                        pred_list.append(pred.data.item())
                        adv_pred_list.append(pred2.data.item())

                    if find_majority(pred_list)[0]==target.data.item():
                        test_acc += 1

                    if find_majority(adv_pred_list)[0]==target.data.item():
                        test_adv_acc += 1

                test_loss /= len(testset.dataset) * args.batch_size
                test_acc /= len(testset.dataset) 

                test_adv_loss /= len(testset.dataset) * args.batch_size
                test_adv_acc /= len(testset.dataset)

                print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                if test_acc > best_test_acc:
                    hypergan.save_models(args, test_acc)
                    best_test_acc = test_acc

                print ('Test Adv Accuracy: {}, Test Adv Loss: {}'.format(test_adv_acc, test_adv_loss))
                if test_adv_loss < best_test_adv_loss:
                    best_test_adv_loss = test_adv_loss
                if test_adv_acc > best_test_adv_acc:
                    best_test_adv_acc = test_adv_acc
                    torch.save(netAttacker.state_dict(), 'saved_models/{}/netAttacker-{}.pt'.format(args.dataset,best_test_adv_acc))
                    
       
if __name__ == '__main__':
    args = load_args()
    train(args)
