import torch
import torch.nn as nn
import torch.nn.functional as F
from .hypergan_base import HyperGAN_Base

""" MedNet Pytorch definition """
class MedNet(nn.Module):
    def __init__(self,args):
        super(MedNet, self).__init__()

        self.num_classes = int(args.num_classes)
        self.after_conv_size = int((args.input_size/4-3)**2*128)
        self.in_channels = int(args.in_channels)
        
        self.conv1 = nn.Conv2d(self.in_channels, 64, 3, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False)
        self.conv4 = nn.Conv2d(128, 128, 3, bias=False)
        self.linear1 = nn.Linear(self.after_conv_size, 256, bias=False)
        self.linear2 = nn.Linear(256, 256, bias=False)
        self.linear3 = nn.Linear(256, self.num_classes, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        # x = x.view(x.size(0), -1)
        x = x.view(-1,self.after_conv_size)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.s, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, self.z*self.ngen, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = x.view(-1, self.s) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        return x
        # x = x.view(-1, self.ngen, self.z)
        # w = torch.stack([x[:, i] for i in range(self.ngen)])
        # return w


class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, 32*3*3*3 + 32, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :32*3*3*3], x[:, -32:]
        w = w.view(-1, 32, 3, 3, 3)
        b = b.view(-1, 32)
        return (w, b)


class GeneratorW2(nn.Module):
    def __init__(self, args):
        super(GeneratorW2, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, 64*32*3*3+64, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :64*32*3*3], x[:, -64:]
        w = w.view(-1, 64, 32, 3, 3)
        b = b.view(-1, 64)
        return (w, b)


class GeneratorW3(nn.Module):
    def __init__(self, args):
        super(GeneratorW3, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, 64*64*3*3+64, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :64*64*3*3], x[:, -64:]
        w = w.view(-1, 64, 64, 3, 3)
        b = b.view(-1, 64)
        return (w, b)


class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, 128*9216+128, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :128*9216], x[:, -128:]
        w = w.view(-1, 128, 9216)
        b = b.view(-1, 128)
        return (w, b)


class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.num_classes = int(args.num_classes)
        self.linear1 = nn.Linear(self.z, 512, bias=self.bias)
        self.linear2 = nn.Linear(512, 512, bias=self.bias)
        self.linear3 = nn.Linear(512, self.num_classes*128+self.num_classes, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :self.num_classes*128], x[:, -self.num_classes:]
        w = w.view(-1, self.num_classes, 128)
        b = b.view(-1, self.num_classes)
        return (w, b)



class HyperGAN(HyperGAN_Base):
    
    def __init__(self, args):
        super(HyperGAN, self).__init__(args)
        self.mixer = Mixer(args).to(args.device)
        self.generator = self.Generator(args)
        self.model = MedNet(args).to(args.device)

    class Generator(object):
        def __init__(self, args):
            self.W1 = GeneratorW1(args).to(args.device)
            self.W2 = GeneratorW2(args).to(args.device)
            self.W3 = GeneratorW3(args).to(args.device)
            self.W4 = GeneratorW4(args).to(args.device)
            self.W5 = GeneratorW5(args).to(args.device)

        def __call__(self, x):
            w1, b1 = self.W1(x[0])
            w2, b2 = self.W2(x[1])
            w3, b3 = self.W3(x[2])
            w4, b4 = self.W4(x[3])
            w5, b5 = self.W5(x[4])
            layers = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5]
            return layers
        
        def as_list(self):
            return [self.W1, self.W2, self.W3, self.W4, self.W5]

    """ functional model for training """
    def eval_f(self, args, Z, data):
        w1, b1, w2, b2, w3, b3, w4, b4, w5, b5 = Z
        x = F.relu(F.conv2d(data, w1, bias=b1))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(F.conv2d(x, w2, stride=1, bias=b2))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(F.conv2d(x, w3, bias=b3))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(F.linear(x, w4, bias=b4))
        x = F.linear(x, w5, bias=b5)
        return x

    def restore_models(self, args):
        d = torch.load(args.resume)
        self.mixer.load_state_dict(d['mixer']['state_dict'])
        generators = self.generator.as_list()
        for i, gen in enumerate(generators):
            gen.load_state_dict(d['W{}'.format(i+1)]['state_dict'])


    def save_models(self, args, metrics=None):
        save_dict = {
                'mixer': {'state_dict': self.mixer.state_dict()},
                'W1': {'state_dict': self.generator.W1.state_dict()},
                'W2': {'state_dict': self.generator.W2.state_dict()},
                'W3': {'state_dict': self.generator.W3.state_dict()},
                'W4': {'state_dict': self.generator.W4.state_dict()},
                'W5': {'state_dict': self.generator.W5.state_dict()}
                }
        path = 'saved_models/{}/mednet-{}.pt'.format(args.dataset, metrics)
        torch.save(save_dict, path)
