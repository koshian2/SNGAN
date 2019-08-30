import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import GeneratorResidualBlock, DiscriminatorSNResidualBlock, SNEmbedding
from models.core_layers import ConditionalBatchNorm2d

class Generator(nn.Module):
    def __init__(self, n_classes_g=0):
        super().__init__()
        self.dense = nn.Linear(128, 3 * 3 * 1024)
        self.block1 = GeneratorResidualBlock(1024, 1024, 2, n_classes=n_classes_g)
        self.block2 = GeneratorResidualBlock(1024, 512, 2, n_classes=n_classes_g)
        self.block3 = GeneratorResidualBlock(512, 256, 2, n_classes=n_classes_g)
        self.block4 = GeneratorResidualBlock(256, 128, 2, n_classes=n_classes_g)
        self.block5 = GeneratorResidualBlock(128, 64, 2, n_classes=n_classes_g)
        self.bn_out = ConditionalBatchNorm2d(64, n_classes_g) if n_classes_g > 0 else nn.BatchNorm2d(64)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 1024, 3, 3)
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x, y), y), y), y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, n_classes_d=0):
        super().__init__()
        self.block1 = DiscriminatorSNResidualBlock(3, 64, 2)
        self.block2 = DiscriminatorSNResidualBlock(64, 128, 2)
        self.block3 = DiscriminatorSNResidualBlock(128, 256, 2)
        self.block4 = DiscriminatorSNResidualBlock(256, 512, 2)
        self.block5 = DiscriminatorSNResidualBlock(512, 1024, 2)
        self.block6 = DiscriminatorSNResidualBlock(1024, 1024, 1)
        self.dense = nn.Linear(1024, 1)
        if n_classes_d > 0:
            self.sn_embedding = SNEmbedding(n_classes_d, 1024)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.block6(self.block5(self.block4(self.block3(self.block2(self.block1(inputs))))))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x

