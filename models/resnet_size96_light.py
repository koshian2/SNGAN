import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import DiscriminatorSNResidualBlock, SNEmbedding
from models.core_layers import ConditionalBatchNorm2d, SpectralNorm


class DiscriminatorLight(nn.Module):
    def __init__(self, n_classes_d=0):
        super().__init__()
        self.initial_conv = SpectralNorm(nn.Conv2d(3, 32, 4, 2, padding=1))
        self.block1 = DiscriminatorSNResidualBlock(32, 64, 2)
        self.block2 = DiscriminatorSNResidualBlock(64, 128, 2)
        self.block3 = DiscriminatorSNResidualBlock(128, 256, 2)
        self.block4 = DiscriminatorSNResidualBlock(256, 512, 2)
        self.block5 = DiscriminatorSNResidualBlock(512, 1024, 1)
        self.dense = nn.Linear(1024, 1)
        if n_classes_d > 0:
            self.sn_embedding = SNEmbedding(n_classes_d, 1024)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.initial_conv(inputs)
        x = self.block5(self.block4(self.block3(self.block2(self.block1(x)))))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x