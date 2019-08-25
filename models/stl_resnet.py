import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import GeneratorResidualBlock, DiscriminatorSNResidualBlock, SNEmbedding
from models.core_layers import ConditionalBatchNorm2d

class Generator(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.dense = nn.Linear(128, 6 * 6 * 512)
        self.block1 = GeneratorResidualBlock(512, 256, 2, n_classes=n_classes)
        self.block2 = GeneratorResidualBlock(256, 128, 2, n_classes=n_classes)
        self.block3 = GeneratorResidualBlock(128, 64, 2, n_classes=n_classes)
        self.bn_out = ConditionalBatchNorm2d(64, n_classes) if enable_conditional else nn.BatchNorm2d(64)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 512, 6, 6)
        x = self.block3(self.block2(self.block1(x, y), y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, enable_conditional=False, last_ch=1024):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.block1 = DiscriminatorSNResidualBlock(3, 64, 2)
        self.block2 = DiscriminatorSNResidualBlock(64, 128, 2)
        self.block3 = DiscriminatorSNResidualBlock(128, 256, 2)
        self.block4 = DiscriminatorSNResidualBlock(256, 512, 2)
        # The last channel is 1024 in the original. But change to 512 for computational cost reasons
        self.block5 = DiscriminatorSNResidualBlock(512, last_ch, 1)
        self.dense = nn.Linear(last_ch, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, last_ch)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.block5(self.block4(self.block3(self.block2(self.block1(inputs)))))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x

