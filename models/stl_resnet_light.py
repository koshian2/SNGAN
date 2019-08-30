import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import DiscriminatorSNResidualBlock, SNEmbedding
from models.core_layers import ConditionalBatchNorm2d, SpectralNorm

class DiscriminatorStrided(nn.Module):
    def __init__(self, enable_conditional=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.initial_down = SpectralNorm(nn.Conv2d(3, 32, 4, 2, padding=0)) #(24,24)
        self.block1 = DiscriminatorSNResidualBlock(32, 64, 2)
        self.block2 = DiscriminatorSNResidualBlock(64, 128, 2)
        self.block3 = DiscriminatorSNResidualBlock(128, 256, 2)
        self.block4 = DiscriminatorSNResidualBlock(256, 512, 1)
        self.dense = nn.Linear(512, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, 512)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.initial_down(inputs)
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x
