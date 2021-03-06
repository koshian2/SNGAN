import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module_postact import GeneratorResidualBlockPostAct, DiscriminatorResidualBlockPostAct
from models.basic_module import ConvSNLRelu, SNEmbedding
from models.core_layers import ConditionalBatchNorm2d

class Generator(nn.Module):
    def __init__(self, latent_dims=3, n_classes_g=0):
        super().__init__()
        self.latent_dims = latent_dims
        self.dense = nn.Linear(128, latent_dims * latent_dims * 512)
        self.block11 = GeneratorResidualBlockPostAct(512, 512, 2, n_classes=n_classes_g)
        self.block12 = GeneratorResidualBlockPostAct(512, 512, 1, n_classes=n_classes_g)

        self.block21 = GeneratorResidualBlockPostAct(512, 256, 2, n_classes=n_classes_g)
        self.block22 = GeneratorResidualBlockPostAct(256, 256, 1, n_classes=n_classes_g)

        self.block31 = GeneratorResidualBlockPostAct(256, 128, 2, n_classes=n_classes_g)
        self.block32 = GeneratorResidualBlockPostAct(128, 128, 1, n_classes=n_classes_g)

        self.block41 = GeneratorResidualBlockPostAct(128, 64, 2, n_classes=n_classes_g)
        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 512, self.latent_dims, self.latent_dims)
        x = self.block12(self.block11(x, y), y)
        x = self.block22(self.block21(x, y), y)
        x = self.block32(self.block31(x, y), y)
        x = self.block41(x, y)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, latent_dims=3, n_classes=0, lrelu_slope=0.1):
        super().__init__()
        self.lrelu_slope = lrelu_slope
        self.conv1 = self.discriminator_block(3, 32)
        self.conv2 = self.discriminator_block(32, 64)
        self.conv3 = self.discriminator_block(64, 128)
        self.conv4 = self.discriminator_block(128, 256)
        self.dense = nn.Linear(latent_dims * latent_dims * 256, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, latent_dims * latent_dims * 256)
        else:
            self.sn_embedding = None

    def discriminator_block(self, in_ch, out_ch):
        return nn.Sequential(
            ConvSNLRelu(in_ch, out_ch, 3, 2, padding=1, lrelu_slope=self.lrelu_slope),
            ConvSNLRelu(out_ch, out_ch, 3, 1, padding=1, lrelu_slope=self.lrelu_slope)
        )

    def forward(self, inputs, label_onehots=None):
        x = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
        base_feature = x.view(inputs.size(0), -1)
        x = self.dense(base_feature)
        if self.sn_embedding is not None:
            x = self.sn_embedding(base_feature, x, label_onehots)
        return x

class DiscriminatorResNet(nn.Module):
    def __init__(self, latent_dims=3, n_classes_d=0, initial_ch=32):
        super().__init__()
        ratio = [1, 1, 2, 2, 4, 4, 8, 8]
        layers = []
        for i, r in enumerate(ratio):
            layers.append(DiscriminatorResidualBlockPostAct(
                3 if i == 0 else ratio[i - 1]*initial_ch, # in_ch
                ratio[i]*initial_ch, # out_ch
                2 if i % 2 == 0 else 1,                
            ))
        self.blocks = nn.Sequential(*layers)
        self.dense = nn.Linear(initial_ch * 8, 1)        
        if n_classes_d > 0:
            self.sn_embedding = SNEmbedding(n_classes_d, initial_ch * 8)            
        else:
            self.sn_embedding = None
            
    def forward(self, inputs, y=None):
        x = self.blocks(inputs)
        features = torch.sum(x, dim=(2, 3))  # global sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x







