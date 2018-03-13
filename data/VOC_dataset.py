import os.path
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import h5py
import time
from data.base_dataset import *
from PIL import Image
import math, random


def make_dataset_fromlst(listfilename):
    """
    NYUlist format:
    imagepath seglabelpath depthpath HHApath
    """
    images = []
    segs = []

    with open(listfilename) as f:
        content = f.readlines()
        for x in content:
            imgname, segname = x.strip().split(' ')
            images += [imgname]
            segs += [segname]

    return {'images':images, 'segs':segs}

class VOCDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(8964)
        self.paths_dict = make_dataset_fromlst(opt.list)
        self.len = len(self.paths_dict['images'])
        self.datafile = 'VOC_dataset.py'

    def __getitem__(self, index):
        #self.paths['images'][index]
        # print self.opt.scale,self.opt.flip,self.opt.crop,self.opt.colorjitter
        img = np.asarray(Image.open(self.paths_dict['images'][index]))
        seg = np.asarray(Image.open(self.paths_dict['segs'][index])).astype(np.uint8)
        # print(np.unique(seg))

        params = get_params(self.opt, seg.shape)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)
        return {'image':img_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'VOCDataset'

class VOCDataset_val(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.paths_dict = make_dataset_fromlst(opt.vallist)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        seg = np.asarray(Image.open(self.paths_dict['segs'][index])).astype(np.uint8)

        params = get_params(self.opt, seg.shape, test=True)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)

        return {'image':img_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'VOCDataset_val'


