import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

class FAC(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FAC, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)


    def forward(self, x):
        x = self.conv(x)
        feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = self.conv_atten(feat)
        feat = self.bn_atten(feat)
        feat = feat.sigmoid()
        out = torch.mul(x, feat)
        return out

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x