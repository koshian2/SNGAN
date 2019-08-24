import torch
from torch import nn
import torch.nn.functional as F
from models.core_layers import SpectralNorm, ConditionalBatchNorm2d

def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

## Discriminator block

class DeconvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0, n_classes=0):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding=padding)
        if n_classes == 0:
            self.bn = nn.BatchNorm2d(out_ch)
        else:
            self.bn = ConditionalBatchNorm2d(out_ch, n_classes)
        self.relu = nn.ReLU(True)

        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots=None):
        x = self.conv(inputs)
        if label_onehots is not None:
            x = self.bn(x, label_onehots)
        else:
            x = self.bn(x)
        return self.relu(x)

class GeneratorResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling, n_classes=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.upsampling = upsampling
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            self.bn1 = ConditionalBatchNorm2d(in_ch, n_classes)
            self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)
        if in_ch != out_ch or upsampling > 1:
            self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut_conv = None

    def forward(self, inputs, label_onehots=None):
        # main
        if label_onehots is not None:
            x = self.bn1(inputs, label_onehots)
        else:
            x = self.bn1(inputs)
        x = F.relu(x)

        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling)
        x = self.conv1(x)

        if label_onehots is not None:
            x = self.bn2(x, label_onehots)
        else:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)

        # short cut
        if self.upsampling > 1:
            shortcut = F.interpolate(inputs, scale_factor=self.upsampling)
        else:
            shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        # residual add
        return x + shortcut
        
## Discriminator Block
class ConvSNLRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0):
        super().__init__()
        self.conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel, stride, padding=padding))
        self.lrelu = nn.LeakyReLU(0.1, True)
        
        self.conv.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.lrelu(self.conv(inputs))

class SNEmbedding(nn.Module):
    def __init__(self, n_classes, out_dims):
        super().__init__()
        self.linear = SpectralNorm(nn.Linear(n_classes, out_dims, bias=False))

        self.linear.apply(init_xavier_uniform)

    def forward(self, output_logits, label_onehots):
        wy = self.linear(label_onehots)
        weighted = torch.sum(output_logits * wy, dim=1, keepdim=True)
        return output_logits + weighted

class DiscriminatorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsampling, n_classes=0):
        super().__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.downsampling = downsampling
        if in_ch != out_ch or downsampling > 1:
            self.shortcut_conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        else:
            self.shortcut_conv = None

    def forward(self, inputs):
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # short cut
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs
        if self.downsampling > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsampling)
            shortcut = F.avg_pool2d(shortcut, kernel_size=self.downsampling)
        # residual add
        return x + shortcut

            

