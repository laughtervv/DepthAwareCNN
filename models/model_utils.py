import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
import time

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6, gamma=1.,beta=0.,learnable=False):
        super(LayerNorm,self).__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.ones(features))
            self.beta = nn.Parameter(torch.zeros(features))
        else:
            self.gamma = gamma
            self.beta = beta

        self.eps = eps

    def forward(self, x):
        x_size = x.size()
        mean = x.view(x_size[0],x_size[1],x_size[2]*x_size[3]).mean(2)\
            .view(x_size[0],x_size[1],1,1).repeat(1, 1, x_size[2], x_size[3])
        std = x.view(x_size[0],x_size[1],x_size[2]*x_size[3]).std(2)\
            .view(x_size[0],x_size[1],1,1).repeat(1, 1, x_size[2], x_size[3])
        # print 'mean',mean.size(),'x',x_size
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class CaffeNormalize(nn.Module):

    def __init__(self, features, eps=1e-7):
        super(CaffeNormalize,self).__init__()
        self.scale = nn.Parameter(10.*torch.ones(features))#, requires_grad=False)
        self.eps = eps

    def forward(self, x):
        # print self.scale
        x_size = x.size()
        norm = x.norm(2,dim=1,keepdim=True)#.detach()
        #print norm.data.cpu().numpy(),self.scale.mean().data.cpu().numpy()#,self.scale.grad.mean().data.cpu().numpy()
        x = x.div(norm+self.eps)

        return x.mul(self.scale.view(1, x_size[1], 1, 1))


class DepthGlobalPool(nn.Module):
    def __init__(self, n_features, n_out):
        super(DepthGlobalPool, self).__init__()
        self.model = nn.Conv2d(n_features, n_out, kernel_size=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.norm = CaffeNormalize(n_out)
        self.dropout = nn.Dropout(0.3)

        n = self.model.kernel_size[0] * self.model.kernel_size[1] * self.model.out_channels
        self.model.weight.data.normal_(0, np.sqrt(2. / n))
        if self.model.bias is not None:
            self.model.bias.data.zero_()

    def forward(self, features, depth, depthpool=False):
        # features = self.pool(self.model(features))
        out2_size = features.size()
        features = self.model(features)

        if isinstance(depth, Variable) and depthpool:
            outfeatures = features.clone()
            n_c = features.size()[1]

            # depth-wise average pooling
            # depthclone = depth.clone()
            depth = depth.data.cpu().numpy()
            _, depth_bin = np.histogram(depth)

            bin_low = depth_bin[0]
            for bin_high in depth_bin[1:]:
                indices = ((depth <= bin_high) & (depth >= bin_low)).nonzero()
                if indices[0].shape[0] != 0:
                    for j in range(n_c):
                        output_ins = features[indices[0], indices[1] + j, indices[2], indices[3]]
                        mean_feat = torch.mean(output_ins).expand_as(output_ins)
                        outfeatures[indices[0], indices[1] + j, indices[2], indices[3]] = mean_feat  # torch.mean(output_ins)
                    bin_low = bin_high

            # outfeatures = self.norm(outfeatures)
            outfeatures = self.dropout(outfeatures)

            # bin_low = depth_bin[0]
            # for bin_high in depth_bin[1:]:
            #     indices = ((depth <= bin_high) & (depth >= bin_low)).nonzero()
            #     if indices[0].shape[0] != 0:
            #         output_ins = features[indices[0], indices[1], indices[2], indices[3]]
            #         mean_feat = torch.mean(output_ins).expand_as(output_ins)
            #         depthclone[indices[0], indices[1], indices[2], indices[3]] = mean_feat
            #         bin_low = bin_high
            #
            # upsample = nn.UpsamplingBilinear2d(scale_factor=8)
            # torchvision.utils.save_image(upsample(depthclone).data, 'depth_feature1.png', normalize=True, range=(0, 1))
            # outfeatures = self.dropout(outfeatures)
        else:
            features = self.pool(features)
            # features = self.norm(features)
            outfeatures = self.dropout(features)
            self.upsample = nn.UpsamplingBilinear2d((out2_size[2],out2_size[3]))
            outfeatures = self.upsample(outfeatures)

        return outfeatures
