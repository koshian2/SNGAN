import torch
from torch import nn
from core_layers import SpectralNorm, ConditionalBatchNorm2d

def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

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

    

