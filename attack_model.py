import torch.nn as nn

class _netAttacker(nn.Module):
    ''' The attacker model is given an image and outputs a perturbed version of that image.''' 
    def __init__(self, imageSize, imageChannels):
        super(_netAttacker, self).__init__()
        self.imageSize = imageSize
        self.imageChannels = imageChannels
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(100, 32 * 8, 3, 1, 0, bias=True),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(32 * 2, 32, 3, 2, 1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, 2, 1, bias=True),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(3*33*33, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, self.imageChannels*self.imageSize*self.imageSize),
        )
        self.tanh = nn.Sequential(
            nn.Tanh(),
        )
    def forward(self, noise):
        x = self.conv(noise)
        x = x.view(-1, 3*33*33)
        x = self.fc(x)
        x = x.view(-1, self.imageChannels, self.imageSize, self.imageSize)
        return x