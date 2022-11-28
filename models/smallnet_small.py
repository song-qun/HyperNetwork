import torch
import torch.nn as nn
import torch.nn.functional as F
from .hypergan_base import HyperGAN_Base


""" class model of target network for testing """
class Small(nn.Module):
    def __init__(self,args):
        super(Small, self).__init__()

        self.num_classes=int(args.num_classes)
        self.in_channels=int(args.in_channels)
        self.input_size=int(args.input_size)

        self.conv1 = nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(32, 32, 5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.linear = nn.Linear(32*int((self.input_size/4-3)*(self.input_size/4-3)), self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 32*int((self.input_size/4-3)*(self.input_size/4-3)))
        x = self.linear(x)
        return x


class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.s, 64, bias=self.bias)
        self.linear2 = nn.Linear(64, 64, bias=self.bias)
        self.linear3 = nn.Linear(64, self.z*self.ngen, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = x.view(-1, self.s) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        # return x
        x = x.view(-1, self.ngen, self.z)
        w = torch.stack([x[:, i] for i in range(self.ngen)])
        return w


class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.in_channels = int(args.in_channels)
        self.linear1 = nn.Linear(self.z, 64, bias=self.bias)
        self.linear2 = nn.Linear(64, 64, bias=self.bias)
        self.linear3 = nn.Linear(64, 32*5*5*self.in_channels + 32, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        if self.bias:
            self.bn1.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :32*5*5*self.in_channels], x[:, -32:]
        w = w.view(-1, 32, self.in_channels, 5, 5)
        b = b.view(-1, 32)
        return (w, b)

class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 64, bias=self.bias)
        self.linear2 = nn.Linear(64, 64, bias=self.bias)
        self.linear3 = nn.Linear(64, 32*32*5*5+32, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :32*32*5*5], x[:, -32:]
        w = w.view(-1, 32, 32, 5, 5)
        b = b.view(-1, 32)
        return (w, b)


class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 64, bias=self.bias)
        self.linear2 = nn.Linear(64, 64, bias=self.bias)
        self.linear3 = nn.Linear(64, 32*int((self.input_size/4-3)*(self.input_size/4-3))*self.num_classes+self.num_classes, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.elu(self.bn1(self.linear1(x)))
        x = F.elu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :32*int((self.input_size/4-3)*(self.input_size/4-3))*self.num_classes], x[:, -self.num_classes:]
        w = w.view(-1, self.num_classes, 32*int((self.input_size/4-3)*(self.input_size/4-3)))
        b = b.view(-1, self.num_classes)
        return (w, b)



class HyperGAN(HyperGAN_Base):
    
    def __init__(self, args):
        super(HyperGAN, self).__init__(args)
        self.mixer = Mixer(args).to(args.device)
        self.generator = self.Generator(args)
        self.model = Small(args).to(args.device)

    class Generator(object):
        def __init__(self, args):
            self.W1 = GeneratorW1(args).to(args.device)
            self.W2 = GeneratorW2(args).to(args.device)
            self.W3 = GeneratorW3(args).to(args.device)

        def __call__(self, x):
            w1, b1 = self.W1(x[0])
            w2, b2 = self.W2(x[1])
            w3, b3 = self.W3(x[2])
            layers = [w1, b1, w2, b2, w3, b3]
            return layers
        
        def as_list(self):
            return [self.W1, self.W2, self.W3]

    """ functional model for training """
    def eval_f(self, args, Z, data):
        w1, b1 = Z[:2]
        w2, b2 = Z[2:4]
        w3, b3 = Z[4:]
        x = F.conv2d(data, w1, stride=1, bias=b1)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.conv2d(x, w2, stride=1, bias=b2)
        x = F.leaky_relu(x)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.view(-1, 32*int((args.input_size/4-3)*(args.input_size/4-3)))
        x = F.linear(x, w3, bias=b3)
        return x

    def restore_models(self, args):
        d = torch.load(args.resume, map_location=args.device)
        self.mixer.load_state_dict(d['mixer']['state_dict'])
        self.mixer.to(args.device)
        generators = self.generator.as_list()
        for i, gen in enumerate(generators):
            gen.load_state_dict(d['W{}'.format(i+1)]['state_dict'])
            gen.to(args.device)


    def save_models(self, args, metrics=None):
        save_dict = {
                'mixer': {'state_dict': self.mixer.state_dict()},
                'W1': {'state_dict': self.generator.W1.state_dict()},
                'W2': {'state_dict': self.generator.W2.state_dict()},
                'W3': {'state_dict': self.generator.W3.state_dict()}
                }
        
        path = 'saved_models/{}/smallnet_small-{}.pt'.format(args.dataset, metrics)
        torch.save(save_dict, path)