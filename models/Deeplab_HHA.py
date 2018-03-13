import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from .base_model import BaseModel
import numpy as np
from . import losses
import shutil
from utils.util import *
from torch.autograd import Variable
from collections import OrderedDict
from tensorboardX import SummaryWriter
import os
import VGG_Deeplab as VGG_Deeplab

class Deeplab_VGG(nn.Module):
    def __init__(self, num_classes, depthconv=False):
        super(Deeplab_VGG,self).__init__()
        self.Scale = VGG_Deeplab.vgg16(num_classes=num_classes,depthconv=depthconv)

    def forward(self,x, depth=None):
        output = self.Scale(x,depth) # for original scale
        return output

class Deeplab_HHA_Solver(BaseModel):
    def __init__(self, opt, dataset=None):
        BaseModel.initialize(self, opt)
        self.model_rgb = Deeplab_VGG(self.opt.label_nc,self.opt.depthconv)
        self.model_HHA = Deeplab_VGG(self.opt.label_nc,self.opt.depthconv)

        self.model = nn.Sequential(*[self.model_rgb,self.model_HHA])

        if self.opt.isTrain:
            self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()
            # self.optimizer = torch.optim.SGD(
            #     [
            #         {'params': self.model_rgb.Scale.get_1x_lr_params_NOscale(), 'lr': self.opt.lr},
            #         {'params': self.model_rgb.Scale.get_10x_lr_params(), 'lr': 10 * self.opt.lr},
            #         {'params': self.model_rgb.Scale.get_2x_lr_params_NOscale(), 'lr': 2 * self.opt.lr,
            #          'weight_decay': 0.},
            #         {'params': self.model_rgb.Scale.get_20x_lr_params(), 'lr': 20 * self.opt.lr, 'weight_decay': 0.},
            #         {'params': self.model_HHA.Scale.get_1x_lr_params_NOscale(), 'lr': self.opt.lr},
            #         {'params': self.model_HHA.Scale.get_10x_lr_params(), 'lr': 10 * self.opt.lr},
            #         {'params': self.model_HHA.Scale.get_2x_lr_params_NOscale(), 'lr': 2 * self.opt.lr,
            #          'weight_decay': 0.},
            #         {'params': self.model_HHA.Scale.get_20x_lr_params(), 'lr': 20 * self.opt.lr, 'weight_decay': 0.}
            #     ],
            #     lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.wd)
            params_rgb = list(self.model_rgb.Scale.parameters())
            params_HHA = list(self.model_HHA.Scale.parameters())
            self.optimizer = torch.optim.SGD(params_rgb+params_HHA, lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.wd)
            #
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.wd)

            self.old_lr = self.opt.lr
            self.averageloss = []
            # copy scripts
            self.model_path = './models' #os.path.dirname(os.path.realpath(__file__))
            self.data_path = './data' #os.path.dirname(os.path.realpath(__file__))
            shutil.copyfile(os.path.join(self.model_path, 'Deeplab_HHA.py'), os.path.join(self.model_dir, 'Deeplab.py'))
            shutil.copyfile(os.path.join(self.model_path, 'VGG_Deeplab.py'), os.path.join(self.model_dir, 'VGG_Deeplab.py'))
            shutil.copyfile(os.path.join(self.model_path, 'model_utils.py'), os.path.join(self.model_dir, 'model_utils.py'))
            shutil.copyfile(os.path.join(self.data_path, dataset.datafile), os.path.join(self.model_dir, dataset.datafile))
            shutil.copyfile(os.path.join(self.data_path, 'base_dataset.py'), os.path.join(self.model_dir, 'base_dataset.py'))

            self.writer = SummaryWriter(self.tensorborad_dir)
            self.counter = 0

        if not self.isTrain or self.opt.continue_train:
            pretrained_path = ''# if not self.isTrain else opt.load_pretrain

            if self.opt.pretrained_model!='' or (self.opt.pretrained_model_HHA != '' and self.opt.pretrained_model_rgb != ''):
                if self.opt.pretrained_model_HHA != '' and self.opt.pretrained_model_rgb != '':
                    self.load_pretrained_network(self.model_rgb, self.opt.pretrained_model_rgb, self.opt.which_epoch_rgb, False)
                    self.load_pretrained_network(self.model_HHA, self.opt.pretrained_model_HHA, self.opt.which_epoch_HHA, False)
                else:
                    self.load_pretrained_network(self.model_rgb, self.opt.pretrained_model, self.opt.which_epoch, False)
                    self.load_pretrained_network(self.model_HHA, self.opt.pretrained_model, self.opt.which_epoch, False)
                print("successfully loaded from pretrained model with given path!")
            else:
                self.load()
                print("successfully loaded from pretrained model 0!")

        self.model_rgb.cuda()
        self.model_HHA.cuda()
        self.normweightgrad=0.

    def forward(self, data, isTrain=True):
        self.model_rgb.zero_grad()
        self.model_HHA.zero_grad()

        # x, depth = None, label = None
        self.image = Variable(data['image']).cuda()
        self.HHA = Variable(data['HHA']).cuda()
        self.depth = Variable(data['depth']).cuda()
        self.seggt = Variable(data['seg']).cuda()

        input_size = self.image.size()
        self.segpred_rgb = self.model_rgb(self.image, self.depth)
        self.segpred_HHA = self.model_HHA(self.HHA, self.depth)

        self.segpred = 0.5*self.segpred_rgb +0.5*self.segpred_HHA#

        self.segpred = nn.functional.upsample(self.segpred, size=(input_size[2], input_size[3]), mode='bilinear')


        if isTrain:
            self.loss = self.criterionSeg(self.segpred, torch.squeeze(self.seggt,1).long())
            self.averageloss += [self.loss.data[0]]

        segpred = self.segpred.max(1, keepdim=True)[1]
        return self.seggt, segpred


    def backward(self, step, total_step):
        self.loss.backward()
        self.optimizer.step()
        if step % self.opt.iterSize  == 0:
            self.update_learning_rate(step, total_step)
            trainingavgloss = np.mean(self.averageloss)
            if self.opt.verbose:
                print ('  Iter: %d, Loss: %f' % (step, trainingavgloss) )

    def get_visuals(self, step):
        ############## Display results and errors ############
        if self.opt.isTrain:
            self.trainingavgloss = np.mean(self.averageloss)
            if self.opt.verbose:
                print ('  Iter: %d, Loss: %f' % (step, self.trainingavgloss) )
            self.writer.add_scalar(self.opt.name+'/trainingloss/', self.trainingavgloss, step)
            self.averageloss = []

        if self.depth is not None:
            return OrderedDict([('image', tensor2im(self.image.data[0], inputmode=self.opt.inputmode)),
                                ('depth', tensor2im(self.depth.data[0], inputmode='divstd-mean')),
                                ('segpred', tensor2label(self.segpred.data[0], self.opt.label_nc)),
                                ('seggt', tensor2label(self.seggt.data[0], self.opt.label_nc))])

    def update_tensorboard(self, data, step):
        if self.opt.isTrain:
            self.writer.add_scalar(self.opt.name+'/Accuracy/', data[0], step)
            self.writer.add_scalar(self.opt.name+'/Accuracy_Class/', data[1], step)
            self.writer.add_scalar(self.opt.name+'/Mean_IoU/', data[2], step)
            self.writer.add_scalar(self.opt.name+'/FWAV_Accuracy/', data[3], step)

            self.writer.add_scalars(self.opt.name+'/loss', {"train": self.trainingavgloss,
                                                             "val": np.mean(self.averageloss)}, step)

            self.writer.add_scalars('trainingavgloss/', {self.opt.name: self.trainingavgloss}, step)
            self.writer.add_scalars('valloss/', {self.opt.name: np.mean(self.averageloss)}, step)
            self.writer.add_scalars('val_MeanIoU/', {self.opt.name: data[2]}, step)

            file_name = os.path.join(self.save_dir, 'MIoU.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('%f\n' % (data[2]))
            # self.writer.add_scalars('losses/'+self.opt.name, {"train": self.trainingavgloss,
            #                                                  "val": np.mean(self.averageloss)}, step)
            self.averageloss = []

    def save(self, which_epoch):
        self.save_network(self.model_rgb, 'rgb', which_epoch, self.gpu_ids)
        self.save_network(self.model_HHA, 'HHA', which_epoch, self.gpu_ids)

    def load(self):
        self.load_network(self.model_rgb, 'rgb',self.opt.which_epoch)
        self.load_network(self.model_HHA, 'HHA',self.opt.which_epoch)

    def update_learning_rate(self, step, total_step):

        lr = max(self.opt.lr * ((1 - float(step) / total_step) ** (self.opt.lr_power)), 1e-7)
        self.writer.add_scalar('Learning_Rate/', lr, step)

        self.optimizer.param_groups[0]['lr'] = lr
        # self.optimizer.param_groups[1]['lr'] = lr*10
        # self.optimizer.param_groups[2]['lr'] = lr*2
        # self.optimizer.param_groups[3]['lr'] = lr*20
        # self.optimizer.param_groups[4]['lr'] = lr
        # self.optimizer.param_groups[5]['lr'] = lr*10
        # self.optimizer.param_groups[6]['lr'] = lr*2
        # self.optimizer.param_groups[7]['lr'] = lr*20

        # torch.nn.utils.clip_grad_norm(self.model_rgb.Scale.get_1x_lr_params_NOscale(), 10.)
        # torch.nn.utils.clip_grad_norm(self.model_rgb.Scale.get_1x_lr_params_NOscale(), 10.)
        # torch.nn.utils.clip_grad_norm(self.model_HHA.Scale.get_10x_lr_params(), 10.)
        # torch.nn.utils.clip_grad_norm(self.model_HHA.Scale.get_10x_lr_params(), 10.)

        if self.opt.verbose:
            print('     update learning rate: %f -> %f' % (self.old_lr, lr))

        self.old_lr = lr