import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import GeneratorResidualBlock, DiscriminatorSNResidualBlock, SNEmbedding
from models.core_layers import ConditionalBatchNorm2d

class Generator(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.dense = nn.Linear(128, 4 * 4 * 256)
        self.block1 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.block2 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.block3 = GeneratorResidualBlock(256, 256, 2, n_classes=n_classes)
        self.bn_out = ConditionalBatchNorm2d(256, n_classes) if enable_conditional else nn.BatchNorm2d(256)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 256, 4, 4)
        x = self.block3(self.block2(self.block1(x, y), y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.block1 = DiscriminatorSNResidualBlock(3, 128, 2)
        self.block2 = DiscriminatorSNResidualBlock(128, 128, 2)
        self.block3 = DiscriminatorSNResidualBlock(128, 128, 1)
        self.block4 = DiscriminatorSNResidualBlock(128, 128, 1)
        self.dense = nn.Linear(128, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, 512)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.block4(self.block3(self.block2(self.block1(inputs))))
        x = F.relu(x, True)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x




