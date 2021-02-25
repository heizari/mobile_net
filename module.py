import numpy as np
import torch.nn as nn
from torch.nn import functional as F
 
class DepthwiseConv(nn.Module):
    def __init__(self, channel_in, kernel_size=3, stride=1, padding=1):
        self.conv = nn.conv2d(channel_in, channel_in, kernel_size=kernel_size,
                              stride=stride, padding=padding, groups=channel_in)
        self.norm = nn.BatchNorm2d(channel_in)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.ReLU(self.norm(x))
        return x

class PointWiseConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=1,
                 stride=1, padding=0):
        self.conv = nn.conv2d(channel_in, channel_out, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        x = self.conv(x)
        x = F.ReLU(self.norm(x))
        return x