import torch
from torch import nn
from models.basic_module import DeconvBNRelu, ConvSNLRelu, SNEmbedding

class Generator(nn.Module):
    def __init__(self, dataset="cifar", enable_conditional=False):
        super().__init__()
        if dataset in ["cifar", "svhn"]:
            self.mg = 4
        elif dataset == "stl":
            self.mg = 6
        if enable_conditional:
            n_classes = 10
        else:
            n_classes = 0

        self.dense = nn.Linear(128, self.mg * self.mg * 512)
        self.conv1 = DeconvBNRelu(512, 256, 4, 2, padding=1, n_classes=n_classes)
        self.conv2 = DeconvBNRelu(256, 128, 4, 2, padding=1, n_classes=n_classes)
        self.conv3 = DeconvBNRelu(128, 64, 4, 2, padding=1, n_classes=n_classes)
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()            
        )

    def forward(self, inputs, label_onehots=None):
        x = self.dense(inputs).view(inputs.size(0), 512, self.mg, self.mg)
        x = self.conv3(self.conv2(self.conv1(x, label_onehots), label_onehots), label_onehots)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, dataset="cifar", enable_conditional=False):
        super().__init__()
        if dataset in ["cifar", "svhn"]:
            self.mg = 4
        elif dataset == "stl":
            self.mg = 6
        if enable_conditional:
            n_classes = 10
        else:
            n_classes = 0

        self.conv1 = self.discriminator_block(3, 64)
        self.conv2 = self.discriminator_block(64, 128)
        self.conv3 = self.discriminator_block(128, 256)
        self.conv4 = ConvSNLRelu(256, 512, 3, 1, padding=1)
        self.dense = nn.Linear(self.mg * self.mg * 512, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, 512)
        else:
            self.sn_embedding = None

    def discriminator_block(self, in_ch, out_ch):
        return nn.Sequential(
            ConvSNLRelu(in_ch, out_ch, 3, 1, padding=1),
            ConvSNLRelu(out_ch, out_ch, 4, 2, padding=1)
        )

    def forward(self, inputs, label_onehots=None):
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        x = self.dense(x.view(inputs.size(0), -1))
        if self.sn_embedding is not None:
            x = self.sn_embedding(x, label_onehots)
        return x
