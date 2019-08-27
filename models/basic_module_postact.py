import torch
from torch import nn
import torch.nn.functional as F
from models.core_layers import SpectralNorm, ConditionalBatchNorm2d
from models.basic_module import init_xavier_uniform

## Generator ResBlockのPost Activation version(original = pre-act)
class GeneratorResidualBlockPostAct(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling, n_classes=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.upsampling = upsampling
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(out_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            self.bn1 = ConditionalBatchNorm2d(out_ch, n_classes)
            self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)
        if in_ch != out_ch or upsampling > 1:
            self.shortcut_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs, label_onehots=None):
        # main
        if self.upsampling > 1:
            x = F.interpolate(inputs, scale_factor=self.upsampling)
        else:
            x = inputs
        x = self.conv1(x)

        if label_onehots is not None:
            x = self.bn1(x, label_onehots)
        else:
            x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)

        if label_onehots is not None:
            x = self.bn2(x, label_onehots)
        else:
            x = self.bn2(x)
        x = F.relu(x)

        # short cut
        if self.upsampling > 1:
            shortcut = F.interpolate(inputs, scale_factor=self.upsampling)
        else:
            shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        # residual add
        return x + shortcut

class DiscriminatorResidualBlockPostAct(nn.Module):
    def __init__(self, in_ch, out_ch, downsampling):
        super().__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        self.conv2 = SpectralNorm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.downsampling = downsampling
        if in_ch != out_ch or downsampling > 1:
            self.shortcut_conv = SpectralNorm(nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0))
        else:
            self.shortcut_conv = None

        self.conv1.apply(init_xavier_uniform)
        self.conv2.apply(init_xavier_uniform)

    def forward(self, inputs):
        if self.downsampling > 1:
            x = F.avg_pool2d(inputs, kernel_size=self.downsampling)
            shortcut = F.avg_pool2d(inputs, kernel_size=self.downsampling)
        else:
            x = inputs
            shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        x = F.relu(self.conv2(F.relu(self.conv1(x))))
        # residual add
        return x + shortcut
