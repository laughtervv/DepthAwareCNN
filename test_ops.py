import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple
from torch.autograd import Variable

from utils.gradcheck import gradcheck
from models.ops.depthconv.functions.depthconv import DepthconvFunction


N, inC, inH, inW = 4, 2, 8, 8
kH, kW = 3, 3
pad, stride, dilation = 0, 1, 1

offC = 1 * 2 * kH * kW

outC = 1
outH = (inH + 2 * pad - (dilation * (kH - 1) + 1)) // stride + 1
outW = (inW + 2 * pad - (dilation * (kW - 1) + 1)) // stride + 1

conv_offset2d = DepthconvFunction(
        padding=(pad, pad),
        stride=(stride, stride),
        dilation=(dilation, dilation), bias=True)
conv2d = F.ConvNd(_pair(stride), _pair(pad), _pair(dilation), False,
               _pair(0), 1, torch.backends.cudnn.benchmark, torch.backends.cudnn.enabled)
offset = Variable(
        torch.ones(N, 1, inH, inW).cuda(),
        requires_grad=False)
input = Variable(
        torch.rand(N, inC, inH, inW).cuda(),
        requires_grad=True)
input2 = Variable(input.data.clone(),
        requires_grad=True)
weight = Variable(
        10*torch.rand(outC, inC, kH, kW).cuda(),
        requires_grad=True)
weight2 = Variable(weight.data.clone(),
        requires_grad=True)
bias = Variable(torch.rand(outC).cuda(),requires_grad=True)
bias2 = Variable(bias.data.clone(),
        requires_grad=True)
grad = Variable(
        torch.rand(N, outC, 6, 6).cuda(),
        requires_grad=True)

print bias
out1 = conv_offset2d(input, offset, weight, bias)
out2 = conv2d(input2, weight2, bias2)
print (out1-out2).sum()

out1.backward(grad)
out2.backward(grad)


print (weight.grad-weight2.grad).sum()
print ('input.grad',input.grad.sum())
print ('input.grad',input2.grad.sum())
print (input.grad-input2.grad).sum()
print (bias.grad-bias2.grad).sum()


# print bias.data.cpu().numpy().dtype

# print("pass gradcheck: {}".format(gradcheck(conv_offset2d, (input, offset, weight, bias))))
# print("pass gradcheck: {}".format(gradcheck(conv2d, (input, weight,None))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ops.depthavgpooling.functions.depthavgpooling import DepthavgpoolingFunction
from models.ops.depthavgpooling.modules import Depthavgpooling
from torch.autograd import Variable

depth = [[[1,0,1,10000],
         [0,1,10000,1],
         [1,0,1,0],
         [0,1,0,1]],
         ]
depth = np.zeros([40,40])
inputarray = torch.Tensor(np.asarray(range(2*40*40)).reshape([1,2,40,40]))
depth = torch.Tensor(np.asarray(depth).reshape([1,1,40,40]))

print inputarray
N, inC, inH, inW = 4, 512, 50, 65
input = Variable(
        inputarray,
        requires_grad=True).cuda()
depth = Variable(
    depth,
        requires_grad=True).cuda()
kH, kW = 3, 3
pad, stride, dilation = 1, 1, 1
depthpooling = Depthavgpooling(kH,stride,pad)
pooling = nn.AvgPool2d(kernel_size=kH, stride=stride,padding=pad)

out1 = depthpooling(input, depth)
out2 = pooling(input)

grad = Variable(
        torch.ones(N, 2, 40, 40).cuda(),
        requires_grad=True)
out1.backward(grad)

print out1-out2
