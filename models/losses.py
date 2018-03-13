import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from distutils.version import LooseVersion

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input).cuda()
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1).cuda()
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    # mask = (target != 255)
    # target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False, ignore_index=255).cuda()
    if size_average:
        loss /= (n*h*w)
    return loss

class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2., weight=None, size_average=True, ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.log_softmax(inputs), targets)


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num+1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):  # variables
        P = F.softmax(inputs)

        b,c,h,w = inputs.size()
        class_mask = Variable(torch.zeros([b,c+1,h,w]).cuda())
        class_mask.scatter_(1, targets.long(), 1.)
        class_mask = class_mask[:,:-1,:,:]

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        # print('alpha',self.alpha.size())
        alpha = self.alpha[targets.data.view(-1)].view_as(targets)
        # print (alpha.size(),class_mask.size(),P.size())
        probs = (P * class_mask).sum(1)  # + 1e-6#.view(-1, 1)
        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
