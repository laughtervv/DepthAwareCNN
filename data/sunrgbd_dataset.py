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
    HHAs = []
    with open(listfilename) as f:
        content = f.readlines()
        for x in content:
            imgname, segname, depthname, HHAname = x.strip().split(' ')
            images += [imgname]
            segs += [segname]
            depths += [depthname]
            HHAs += [HHAname]
    return {'images':images, 'segs':segs, 'HHAs':HHAs, 'depths':depths}

class SUNRGBDDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(int(time.time()))
        self.paths_dict = make_dataset_fromlst(opt.list)
        self.len = len(self.paths_dict['images'])
        # self.label_weight = torch.Tensor(label_weight)
        self.datafile = 'sunrgbd_dataset.py'

    def __getitem__(self, index):
        #self.paths['images'][index]
        # print self.opt.scale,self.opt.flip,self.opt.crop,self.opt.colorjitter
        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        HHA = np.asarray(Image.open(self.paths_dict['HHAs'][index]))[:,:,::-1]
        seg = np.asarray(Image.open(self.paths_dict['segs'][index])).astype(np.uint8)-1
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.uint16)

        assert (img.shape[0]==HHA.shape[0]==seg.shape[0]==depth.shape[0])
        assert (img.shape[1]==HHA.shape[1]==seg.shape[1]==depth.shape[1])

        depth = np.bitwise_or(np.right_shift(depth,3),np.left_shift(depth,16-3))
        depth = depth.astype(np.float32)/120. # 1/5 * depth




        params = get_params_sunrgbd(self.opt, seg.shape, maxcrop=.7)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, normalize=False, istrain=self.opt.isTrain, option=2)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, istrain=self.opt.isTrain, option=2)


        # print img_tensor_tranformed
        # print(np.unique(depth_tensor_tranformed.numpy()).shape)
        # print img_tensor_tranformed.size()
        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'HHA': HHA_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'sunrgbd_dataset'

class SUNRGBDDataset_val(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(8964)
        self.paths_dict = make_dataset_fromlst(opt.vallist)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        #self.paths['images'][index]
        img = np.asarray(Image.open(self.paths_dict['images'][index]))#.astype(np.uint8)
        HHA = np.asarray(Image.open(self.paths_dict['HHAs'][index]))[:,:,::-1]
        seg = np.asarray(Image.open(self.paths_dict['segs'][index])).astype(np.uint8)-1
        depth = np.asarray(Image.open(self.paths_dict['depths'][index])).astype(np.uint16)
        depth = np.bitwise_or(np.right_shift(depth,3),np.left_shift(depth,16-3))
        depth = depth.astype(np.float32)/120. # 1/5 * depth

        assert (img.shape[0]==HHA.shape[0]==seg.shape[0]==depth.shape[0])
        assert (img.shape[1]==HHA.shape[1]==seg.shape[1]==depth.shape[1])

        params = get_params_sunrgbd(self.opt, seg.shape, test=True)
        depth_tensor_tranformed = transform(depth, params, normalize=False,istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False,method='nearest',istrain=self.opt.isTrain)
        # HHA_tensor_tranformed = transform(HHA, params,istrain=self.opt.isTrain)
        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(img, params, normalize=False, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, normalize=False, istrain=self.opt.isTrain, option=2)
        else:
            img_tensor_tranformed = transform(img, params, istrain=self.opt.isTrain, option=1)
            HHA_tensor_tranformed = transform(HHA, params, istrain=self.opt.isTrain, option=2)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'HHA': HHA_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'sunrgbd_dataset_Val'


