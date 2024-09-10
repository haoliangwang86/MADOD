'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models import VGG16_Weights


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.n_outputs = 4608

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.network = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        del self.network.classifier
        self.network.classifier = Identity()
        self.network.avgpool = self.avgpool

        self.freeze_bn()

    def forward(self, x):
        out = self.network(x)
        return out

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def vgg16():
    return VGG()


def vgg16_cifar100():
    return VGG('VGG16', 100)


def test():
    net = VGG('VGG16', 10)
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())

# test()
