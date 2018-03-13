from model_utils import *
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from .ops.depthconv.modules import DepthConv
from .ops.depthavgpooling.modules import Depthavgpooling
import torch
import torchvision

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


cfg = {
 # name:c1_1 c1_2     c2_1 c2_2      c3_1 c3_2 c3_3      c4_1 c4_2 c4_3      c5_1 c5_2 c5_3
 # dilation:                                                                   2    2    2
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

depth_cfg = {
    'D': [0,3,6,10,14],
}


class ConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 bn=False,
                 maxpool=False, pool_kernel=3, pool_stride=2, pool_pad=1):
        super(ConvModule, self).__init__()
        conv2d = nn.Conv2d(inplanes,planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        if maxpool:
            layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride,padding=pool_pad)]

        self.layers = nn.Sequential(*([conv2d]+layers))
    def forward(self, x):
        # x = self.conv2d(x)
        x = self.layers(x)
        return x

class DepthConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1,bn=False):
        super(DepthConvModule, self).__init__()

        conv2d = DepthConv(inplanes,planes,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*([conv2d]+layers))#(*layers)

    def forward(self, x, depth):

        for im,module in enumerate(self.layers._modules.values()):
            if im==0:
                x = module(x,depth)
            else:
                x = module(x)
        # x = self.conv2d(x, depth)
        # x = self.layers(x)
        return x


class VGG_layer2(nn.Module):

    def __init__(self, batch_norm=False, depthconv=False):
        super(VGG_layer2, self).__init__()
        in_channels = 3
        self.depthconv = depthconv
        # if self.depthconv:
        #     self.conv1_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
        #     self.conv1_1 = DepthConvModule(3, 64, bn=batch_norm)
        # else:
        self.conv1_1 = ConvModule(3, 64, bn=batch_norm)
        self.conv1_2 = ConvModule(64, 64, bn=batch_norm, maxpool=True)

        # if self.depthconv:
        # self.conv2_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
        self.downsample_depth2_1 = nn.AvgPool2d(3,padding=1,stride=2)
        #     self.conv2_1 = DepthConvModule(64, 128, bn=batch_norm)
        # else:
        self.conv2_1 = ConvModule(64, 128, bn=batch_norm)
        self.conv2_2 = ConvModule(128, 128, bn=batch_norm, maxpool=True)

        # if self.depthconv:
        #     self.conv3_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
        self.downsample_depth3_1 = nn.AvgPool2d(3,padding=1,stride=2)
        #     self.conv3_1 = DepthConvModule(128, 256, bn=batch_norm)
        # else:
        self.conv3_1 = ConvModule(128, 256, bn=batch_norm)
        self.conv3_2 = ConvModule(256, 256, bn=batch_norm)
        self.conv3_3 = ConvModule(256, 256, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv4_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.downsample_depth4_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv4_1 = DepthConvModule(256, 512, bn=batch_norm)
        else:
            self.conv4_1 = ConvModule(256, 512, bn=batch_norm)
        self.conv4_2 = ConvModule(512, 512, bn=batch_norm)
        self.conv4_3 = ConvModule(512, 512, bn=batch_norm,
                                  maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)

        if self.depthconv:
            self.conv5_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.conv5_1 = DepthConvModule(512, 512, bn=batch_norm,dilation=2,padding=2)
        else:
            self.conv5_1 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_2 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_3 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2,
                                  maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)
        # self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)

    def forward(self, x, depth=None):
        # print x.size()
        # if self.depthconv:
        #     # print self.conv1_1_depthconvweight
        #     x = self.conv1_1(x,self.conv1_1_depthconvweight * depth)
        # else:
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        # if self.depthconv:
        depth = self.downsample_depth2_1(depth)
        #     x = self.conv2_1(x, self.conv2_1_depthconvweight * depth)
        # else:
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        # if self.depthconv:
        depth = self.downsample_depth3_1(depth)
        #     x = self.conv3_1(x, self.conv3_1_depthconvweight * depth)
        # else:
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.depthconv:
            depth = self.downsample_depth4_1(depth)
            x = self.conv4_1(x, self.conv4_1_depthconvweight * depth)
        else:
            x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.depthconv:
            x = self.conv5_1(x, self.conv5_1_depthconvweight * depth)
        else:
            x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5a(x)
        return x,depth

class VGG_layer(nn.Module):

    def __init__(self, batch_norm=False, depthconv=False):
        super(VGG_layer, self).__init__()
        in_channels = 3
        self.depthconv = depthconv
        if self.depthconv:
            self.conv1_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.conv1_1 = DepthConvModule(3, 64, bn=batch_norm)
        else:
            self.conv1_1 = ConvModule(3, 64, bn=batch_norm)
        self.conv1_2 = ConvModule(64, 64, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv2_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.downsample_depth2_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv2_1 = DepthConvModule(64, 128, bn=batch_norm)
        else:
            self.conv2_1 = ConvModule(64, 128, bn=batch_norm)
        self.conv2_2 = ConvModule(128, 128, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv3_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.downsample_depth3_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv3_1 = DepthConvModule(128, 256, bn=batch_norm)
        else:
            self.conv3_1 = ConvModule(128, 256, bn=batch_norm)
        self.conv3_2 = ConvModule(256, 256, bn=batch_norm)
        self.conv3_3 = ConvModule(256, 256, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv4_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.downsample_depth4_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv4_1 = DepthConvModule(256, 512, bn=batch_norm)
        else:
            self.conv4_1 = ConvModule(256, 512, bn=batch_norm)
        self.conv4_2 = ConvModule(512, 512, bn=batch_norm)
        self.conv4_3 = ConvModule(512, 512, bn=batch_norm,
                                  maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)

        if self.depthconv:
            self.conv5_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.conv5_1 = DepthConvModule(512, 512, bn=batch_norm,dilation=2,padding=2)
        else:
            self.conv5_1 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_2 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_3 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2,
                                  maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)
        self.pool5a_d = Depthavgpooling(kernel_size=3, stride=1,padding=1)

    def forward(self, x, depth=None):
        # print x.size()
        if self.depthconv:
            # print self.conv1_1_depthconvweight
            x = self.conv1_1(x,self.conv1_1_depthconvweight * depth)
        else:
            x = self.conv1_1(x)
        x = self.conv1_2(x)
        if self.depthconv:
            depth = self.downsample_depth2_1(depth)
            x = self.conv2_1(x, self.conv2_1_depthconvweight * depth)
        else:
            x = self.conv2_1(x)
        # print 'xxxxxx',x.size()
        x = self.conv2_2(x)
        if self.depthconv:
            depth = self.downsample_depth3_1(depth)
            x = self.conv3_1(x, self.conv3_1_depthconvweight * depth)
        else:
            x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.depthconv:
            depth = self.downsample_depth4_1(depth)
            # print (depth.mean(),depth.max(),depth.min())
            # torchvision.utils.save_image(depth.data, 'depth.png')
            x = self.conv4_1(x, self.conv4_1_depthconvweight * depth)
        else:
            x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.depthconv:
            x = self.conv5_1(x, self.conv5_1_depthconvweight * depth)
        else:
            x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        # x = self.pool5a(x,depth)
        if self.depthconv:
            x = self.pool5a_d(x,depth)
        else:
            x = self.pool5a(x)

        return x, depth

def make_layers(cfg, depth_cfg=[], batch_norm=False, depthconv=False):
    layers = []
    in_channels = 3
    for iv, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if depthconv and iv in depth_cfg:
                conv2d = DepthConv(in_channels, v, kernel_size=3, padding=1)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class Classifier_Module(nn.Module):

    def __init__(self, num_classes, inplanes, depthconv=False):
        super(Classifier_Module, self).__init__()
        # [6, 12, 18, 24]
        self.depthconv = depthconv
        if depthconv:
            self.fc6_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.fc6_1 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=6, dilation=6)  # fc6
        else:
            self.fc6_1 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=6, dilation=6)  # fc6

        self.fc7_1 = nn.Sequential(
            *[nn.ReLU(True), nn.Dropout(),
              nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])  # fc7
        self.fc8_1 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

        if depthconv:
            self.fc6_2_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.fc6_2 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6
        else:
            self.fc6_2 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6

        self.fc7_2 = nn.Sequential(
            *[nn.ReLU(True), nn.Dropout(),
              nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])  # fc7
        self.fc8_2 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

        if depthconv:
            self.fc6_3_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.fc6_3 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=18, dilation=18)  # fc6
        else:
            self.fc6_3 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=18, dilation=18)  # fc6

        self.fc7_3 = nn.Sequential(
            *[nn.ReLU(True), nn.Dropout(),
              nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])  # fc7
        self.fc8_3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

        if depthconv:
            self.fc6_4_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.fc6_4 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=24, dilation=24)  # fc6
        else:
            self.fc6_4 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=24, dilation=24)  # fc6

        self.fc7_4 = nn.Sequential(
            *[nn.ReLU(True), nn.Dropout(),
              nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])  # fc7
        self.fc8_4 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

    def forward(self, x, depth=None):
        if self.depthconv:
            out1 = self.fc6_1(x, self.fc6_1_depthconvweight * depth)
        else:
            out1 = self.fc6_1(x)
        out1 = self.fc7_1(out1)
        out1 = self.fc8_1(out1)

        if self.depthconv:
            out2 = self.fc6_2(x, self.fc6_2_depthconvweight * depth)
        else:
            out2 = self.fc6_2(x)
        out2 = self.fc7_2(out2)
        out2 = self.fc8_2(out2)

        if self.depthconv:
            out3 = self.fc6_3(x, self.fc6_3_depthconvweight * depth)
        else:
            out3 = self.fc6_3(x)
        out3 = self.fc7_3(out3)
        out3 = self.fc8_3(out3)

        if self.depthconv:
            out4 = self.fc6_4(x, self.fc6_4_depthconvweight * depth)
        else:
            out4 = self.fc6_4(x)
        out4 = self.fc7_4(out4)
        out4 = self.fc8_4(out4)

        return out1+out2+out3+out4

class Classifier_Module2(nn.Module):

    def __init__(self, num_classes, inplanes, depthconv=False):
        super(Classifier_Module2, self).__init__()
        # [6, 12, 18, 24]
        self.depthconv = depthconv
        if depthconv:
            self.fc6_2_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.fc6_2 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
            self.downsample_depth = None
        else:
            self.downsample_depth = nn.AvgPool2d(9,padding=1,stride=8)
            self.fc6_2 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6

        self.fc7_2 = nn.Sequential(
            *[nn.ReLU(True), nn.Dropout(),
              nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])  # fc7

        # self.globalpooling = DepthGlobalPool(1024,3)#
        # self.fc8_2 = nn.Conv2d(1024+3, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))#nn.AvgPool2d((54,71))#
        self.dropout = nn.Dropout(0.3)
        # self.norm = CaffeNormalize(1024)#LayerNorm(1024)#nn.InstanceNorm2d(1024).use_running_stats(mode=False)
        self.fc8_2 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

    def forward(self, x, depth=None):
        if self.depthconv:
            out2 = self.fc6_2(x, self.fc6_2_depthconvweight * depth)
        else:
            out2 = self.fc6_2(x)
        out2 = self.fc7_2(out2)
        out2_size = out2.size()

        #global pooling
        globalpool = self.globalpooling(out2)
        # globalpool = self.dropout(self.norm(globalpool))
        globalpool = self.dropout(globalpool)#self.norm(globalpool))
        upsample = nn.Upsample((out2_size[2],out2_size[3]), mode='bilinear')#scale_factor=8)
        globalpool = upsample(globalpool)

        #global pooling with depth
        # globalpool = self.globalpooling(out2,depth)


        # print globalpool.mean()
        out2 = torch.cat([out2, globalpool], 1)
        out2 = self.fc8_2(out2)
        # print out2.size()
        return out2

class VGG(nn.Module):

    def __init__(self, num_classes=20, init_weights=True, depthconv=False,bn=False):
        super(VGG, self).__init__()
        self.features = VGG_layer(batch_norm=bn,depthconv=depthconv)
        self.classifier = Classifier_Module2(num_classes,512,depthconv=depthconv)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, depth=None):
        x,depth = self.features(x,depth)
        x = self.classifier(x,depth)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_normalize_params(self):
        b=[]
        b.append(self.classifier.norm)
        for i in b:
            if isinstance(i, CaffeNormalize):
                yield i.scale

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features.conv1_1)
        b.append(self.features.conv1_2)
        b.append(self.features.conv2_1)
        b.append(self.features.conv2_2)
        b.append(self.features.conv3_1)
        b.append(self.features.conv3_2)
        b.append(self.features.conv3_3)
        b.append(self.features.conv4_1)
        b.append(self.features.conv4_2)
        b.append(self.features.conv4_3)
        b.append(self.features.conv5_1)
        b.append(self.features.conv5_2)
        b.append(self.features.conv5_3)
        # b.append(self.classifier.fc6_1)
        b.append(self.classifier.fc6_2)
        # b.append(self.classifier.norm)
        # b.append(self.classifier.fc6_3)
        # b.append(self.classifier.fc6_4)
        # b.append(self.classifier.fc7_1)
        b.append(self.classifier.fc7_2)
        # b.append(self.classifier.fc7_3)
        # b.append(self.classifier.fc7_4)

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.weight.requires_grad:
                        yield j.weight
                elif isinstance(j, DepthConv):
                    if j.weight.requires_grad:
                        yield j.weight


    def get_2x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features.conv1_1)
        b.append(self.features.conv1_2)
        b.append(self.features.conv2_1)
        b.append(self.features.conv2_2)
        b.append(self.features.conv3_1)
        b.append(self.features.conv3_2)
        b.append(self.features.conv3_3)
        b.append(self.features.conv4_1)
        b.append(self.features.conv4_2)
        b.append(self.features.conv4_3)
        b.append(self.features.conv5_1)
        b.append(self.features.conv5_2)
        b.append(self.features.conv5_3)
        # b.append(self.classifier.fc6_1)
        b.append(self.classifier.fc6_2)
        # b.append(self.classifier.fc6_3)
        # b.append(self.classifier.fc6_4)
        # b.append(self.classifier.fc7_1)
        b.append(self.classifier.fc7_2)
        # b.append(self.classifier.globalpooling.model)
        # b.append(self.classifier.fc7_3)
        # b.append(self.classifier.fc7_4)

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias
                elif isinstance(j, DepthConv):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias


    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        # b.append(self.classifier.fc8_1.weight)
        b.append(self.classifier.fc8_2.weight)
        # b.append(self.classifier.globalpooling.model.weight)
        # b.append(self.classifier.fc8_3.weight)
        # b.append(self.classifier.fc8_4.weight)

        for i in b:
            yield i
        # for j in range(len(b)):
        #     for i in b[j]:
        #         yield i

    def get_20x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        # b.append(self.classifier.fc8_1.bias)
        b.append(self.classifier.fc8_2.bias)
        # b.append(self.classifier.globalpooling.model.bias)
        # b.append(self.classifier.fc8_3.bias)
        # b.append(self.classifier.fc8_4.bias)

        for i in b:
            yield i
        # for j in range(len(b)):
        #     for i in b[j]:
        #         yield i

    def get_100x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.features.conv1_1_depthconvweight)
        b.append(self.features.conv2_1_depthconvweight)
        b.append(self.features.conv3_1_depthconvweight)
        b.append(self.features.conv4_1_depthconvweight)
        b.append(self.features.conv5_1_depthconvweight)
        b.append(self.classifier.fc6_1_depthconvweight)
        b.append(self.classifier.fc6_2_depthconvweight)
        b.append(self.classifier.fc6_3_depthconvweight)
        b.append(self.classifier.fc6_4_depthconvweight)

        for j in range(len(b)):
            yield b[j]
            # for i in b[j]:
            #     yield i



def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(bn=False,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(bn=True,**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

