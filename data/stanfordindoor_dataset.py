import os.path
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch
import h5py
from data.base_dataset import *
from PIL import Image
import math, random
import time

def make_dataset_fromlst(listfilename):
    """
    NYUlist format:
    imagepath seglabelpath depthpath HHApath
    """
    images = []
    segs = []
    depths = []
    with open(listfilename) as f:
        content = f.readlines()
        for x in content:
            imgname, segname, depthname = x.strip().split(' ')
            images += [imgname]
            segs += [segname]
            depths += [depthname]
    return {'images':images, 'segs':segs, 'depths':depths}

class StanfordIndoorDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(int(time.time()))
        self.paths_dict = make_dataset_fromlst(opt.list)
        self.len = len(self.paths_dict['images'])
        # self.label_weight = torch.Tensor(label_weight)
        self.datafile = 'stanfordindoor_dataset.py'

    def __getitem__(self, index):

        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.float32)/120. # 1/10 * depth
        seg = np.asarray(Image.open(self.paths_dict['segs'][index]))-1

        params = get_params_sunrgbd(self.opt, seg.shape,maxcrop=0.7, maxscale=1.1)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'stanfordindoor_dataset'

class StanfordIndoorDataset_val(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(8964)
        self.paths_dict = make_dataset_fromlst(opt.vallist)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):

        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.float32)/120. # 1/10 * depth
        seg = np.asarray(Image.open(self.paths_dict['segs'][index]))-1

        params = get_params_sunrgbd(self.opt, seg.shape, test=True)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)

        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'stanfordindoor_dataset'


