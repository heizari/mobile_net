import numpy as np
import torch.nn as nn

class MobileNet(nn.Module):
    def __init__(self, channel_in, channel_out):
        print('do init')
    
    def forward(self, x):
        print('do forward')

model = MobileNet(1,1)