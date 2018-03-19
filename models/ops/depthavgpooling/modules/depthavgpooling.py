import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from ..functions import depth_avgpooling


class Depthavgpooling(Module):
    def __init__(self,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(Depthavgpooling, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

    def forward(self, input, depth):
        return depth_avgpooling(input, depth, self.kernel_size, self.stride, self.padding)
