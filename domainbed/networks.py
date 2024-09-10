# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from domainbed.lib import wide_resnet, wrn, vgg
from domainbed.munit.core.networks import AdaINGen


def load_munit_model(model_path, config_path, reverse=False):
    """Load MUNIT model."""

    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    return MUNITModelOfNatVar(model_path, reverse=reverse, config=config).cuda()


class MUNITModelOfNatVar(nn.Module):
    def __init__(self, fname: str, reverse: bool, config: str):
        """Instantiantion of pre-trained MUNIT model.
        
        Params:
            fname: File name of trained MUNIT checkpoint file.
            reverse: If True, returns model mapping from domain A-->B.
                otherwise, model maps from B-->A.
            config: Configuration .yaml file for MUNIT.
        """

        super(MUNITModelOfNatVar, self).__init__()

        self._config = config
        self._fname = fname
        self._reverse = reverse
        self._gen_A, self._gen_B = self.__load()
        self.delta_dim = self._config['gen']['style_dim']

    def forward(self, x, delta):
        if x.size()[-1] == 32:  # for MNIST
            x = torch.cat([x, torch.zeros(x.size(0), 1, 32, 32).to(x.device)], dim=1)
            orig_content, _ = self._gen_A.encode(x)
            orig_content = orig_content.clone().detach().requires_grad_(False)
            x_out = self._gen_B.decode(orig_content, delta)

            return x_out[:, :2, :, :]
        else:
            orig_content, _ = self._gen_A.encode(x)
            orig_content = orig_content.clone().detach().requires_grad_(False)
            return self._gen_B.decode(orig_content, delta)

    def mixup_ood(self, x, x_neg, delta, gmm, thr, num_classes, n_oods=1):
        if x.size()[-1] == 32:  # for MNIST
            x = torch.cat([x, torch.zeros(x.size(0), 1, 32, 32).to(x.device)], dim=1)
            x_neg = torch.cat([x_neg, torch.zeros(x_neg.size(0), 1, 32, 32).to(x.device)], dim=1)

        self._gen_A.eval()
        self._gen_B.eval()

        orig_content, orig_style = self._gen_A.encode(x)
        orig_content_neg, orig_style_neg = self._gen_A.encode(x_neg)
        orig_content = orig_content.clone().detach().requires_grad_(False)
        orig_content_neg = orig_content_neg.clone().detach().requires_grad_(False)

        ood_probs = torch.ones((x.size(0), num_classes)).to(x.device)
        ood_contents = torch.zeros(orig_content.size()).to(x.device)

        outputs = []
        for _ in range(n_oods):
            while True:
                # use random lambda values
                # random_number_1 = torch.rand(1)[0] * 2 - 1
                # random_number_2 = torch.rand(1)[0] * 2 - 1
                # random_number_1 = torch.rand(1)[0] * 20 - 10
                # random_number_2 = torch.rand(1)[0] * 20 - 10
                # random_number_1 = torch.rand(1)[0] * 200 - 100
                # random_number_2 = torch.rand(1)[0] * 200 - 100
                random_number_1 = torch.rand(1)[0] * 2000 - 1000
                random_number_2 = torch.rand(1)[0] * 2000 - 1000

                # Multiply tensor with random number
                mixup_content = torch.mul(orig_content, random_number_1) + torch.mul(orig_content_neg, random_number_2)

                if x.size()[-1] == 32:  # for MNIST
                    mixup_content_pooled = F.avg_pool2d(mixup_content, kernel_size=8)
                else:
                    mixup_content_pooled = F.avg_pool2d(mixup_content, kernel_size=14)

                mixup_content_pooled = mixup_content_pooled.view(mixup_content_pooled.size(0), -1)

                log_probs = gmm.log_prob(mixup_content_pooled[:, None, :])
                probs = torch.exp(log_probs)

                # find rows in probs that have both values less than threshold
                keep = torch.all(probs < thr, dim=1)

                # assign the rows to ood_probs
                ood_probs[keep] = probs[keep]
                ood_contents[keep] = mixup_content[keep]

                if torch.all(ood_probs < thr):
                    break

            x_out = self._gen_B.decode(ood_contents, delta)  # use random style

            if x.size()[-1] == 32:  # for MNIST
                x_out = x_out[:, :2, :, :]

            outputs.append(x_out)

        outputs = torch.cat(outputs)

        return outputs

    def get_semantic(self, x):
        if x.size()[-1] == 32:  # for MNIST
            x = torch.cat([x, torch.zeros(x.size(0), 1, 32, 32).to(x.device)], dim=1)

        orig_content, _ = self._gen_A.encode(x)
        orig_content = orig_content.clone().detach().requires_grad_(False)

        if x.size()[-1] == 32:  # for MNIST
            orig_content = F.avg_pool2d(orig_content, kernel_size=8)
        else:
            orig_content = F.avg_pool2d(orig_content, kernel_size=14)  # for PACS
            # orig_content = F.avg_pool2d(orig_content, kernel_size=28)  # for VLCS
            # orig_content = F.avg_pool2d(orig_content, kernel_size=16)
        orig_content = orig_content.view(orig_content.size(0), -1)
        return orig_content

    def gen_munit_images(self, x):
        orig_content, orig_style = self._gen_A.encode(x)

        x_reconstruct = self._gen_A.decode(orig_content, orig_style)  # use x's style

        x_random_style = []
        for i in range(6):
            delta = torch.randn(x.size(0), self.delta_dim, 1, 1).to(x.device).requires_grad_(False)

            x_out = self._gen_B.decode(orig_content, delta)  # use random style
            x_random_style.append(x_out)

        return x_reconstruct, x_random_style

    def __load(self):
        """Load MUNIT model from file."""

        def load_munit(fname, letter):
            gen = AdaINGen(self._config[f'input_dim_{letter}'], self._config['gen'])
            gen.load_state_dict(torch.load(fname)[letter])
            return gen.eval()

        gen_A = load_munit(self._fname, 'a')
        gen_B = load_munit(self._fname, 'b')

        if self._reverse is False:
            return gen_A, gen_B  # original order
        return gen_B, gen_A  # reversed order


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth'] - 2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if hparams['resnet18']:
            self.network = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.n_outputs = 512
        else:
            self.network = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.n_outputs = 2048

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()

        self.freeze_bn()
        self.hparams = hparams
        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    elif input_shape[1:3] == (112, 112):  # for backbone experiments
        return wrn.Wide_ResNet(input_shape, 28, 10, hparams['resnet_dropout'])
        # return vgg.vgg16()
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        featurizer = Featurizer(input_shape, hparams)
        classifier = Classifier(
            featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            featurizer, classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)
