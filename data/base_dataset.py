import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size, test=False):
    h, w = size
    if opt.scale and test==False:
        scale = random.uniform(0.76, 1.75)
        new_h = h * scale
        new_w = (new_h * w // h)

        new_h = int(round(new_h / 8) * 8)
        new_w = int(round(new_w / 8) * 8)

    else:
        new_h = h
        new_w = w
        # new_h = int(round(h / 8) * 8)
        # new_w = int(round(w / 8) * 8)

    if opt.flip and test==False:
        flip = random.random() > 0.5
    else:
        flip = False

    crop = False
    x1 = x2 = y1 = y2 = 0
    if opt.crop and test==False:
        # if new_h > 241 and new_w > 321: #424
        if opt.batchSize > 1:
            cropsizeh = 321
            cropsizew = 421#(cropsizeh * new_w // new_h)
        else:
            cropscale = random.uniform(0.6,.9)
            cropsizeh = int (new_h * cropscale)
            cropsizew = int (new_w * cropscale)
            # print cropsizeh,cropsizew,new_h,new_w
        x1 = random.randint(0, np.maximum(0, new_w - cropsizew))
        y1 = random.randint(0, np.maximum(0, new_h - cropsizeh))
        x2 = x1 + cropsizew -1
        y2 = y1 + cropsizeh -1
        crop = True

        # if opt.batchSize > 1:
        #     print cropsizew,cropsizeh
    if opt.colorjitter and test==False:
        colorjitter = True
    else:
        colorjitter = False
    return {'scale': (new_w, new_h),
            'flip': flip,
            'crop_pos': (x1, x2, y1, y2),
            'crop': crop,
            'colorjitter': colorjitter}

def get_params_sunrgbd(opt, size, test=False, maxcrop=0.8, maxscale=1.75):
    h, w = size
    if opt.scale and test==False:
        scale = random.uniform(0.76, maxscale)
        new_h = h * scale
        new_w = (new_h * w // h)

        new_h = int(round(new_h / 8) * 8)
        new_w = int(round(new_w / 8) * 8)

    else:
        new_h = h
        new_w = w
        # new_h = int(round(h / 8) * 8)
        # new_w = int(round(w / 8) * 8)

    if opt.flip and test==False:
        flip = random.random() > 0.5
    else:
        flip = False

    crop = False
    x1 = x2 = y1 = y2 = 0
    if opt.crop and test==False:
        # if new_h > 241 and new_w > 321: #424
        if opt.batchSize > 1:
            cropsizeh = 321
            cropsizew = 421#(cropsizeh * new_w // new_h)
        else:
            cropscale = random.uniform(0.6,maxcrop)
            cropsizeh = int (new_h * cropscale)
            cropsizew = int (new_w * cropscale)
            # print cropsizeh,cropsizew,new_h,new_w
        x1 = random.randint(0, np.maximum(0, new_w - cropsizew))
        y1 = random.randint(0, np.maximum(0, new_h - cropsizeh))
        x2 = x1 + cropsizew -1
        y2 = y1 + cropsizeh -1
        crop = True

        # if opt.batchSize > 1:
        #     print cropsizew,cropsizeh
    if opt.colorjitter and test==False:
        colorjitter = True
    else:
        colorjitter = False
    return {'scale': (new_w, new_h),
            'flip': flip,
            'crop_pos': (x1, x2, y1, y2),
            'crop': crop,
            'colorjitter': colorjitter}

def transform(numpyarray, params, normalize=True, method='linear', istrain=True, colorjitter=False, option=0):
    # print params['crop'],params['colorjitter'],params['flip']
    if method == 'linear':
        numpyarray = cv2.resize(numpyarray, (params['scale'][0], params['scale'][1]), interpolation=cv2.INTER_LINEAR)
    else:
        numpyarray = cv2.resize(numpyarray, (params['scale'][0], params['scale'][1]), interpolation=cv2.INTER_NEAREST)

    if istrain:
        if params['crop']:
            # print (numpyarray.shape,params['crop_pos'])
            numpyarray = numpyarray[params['crop_pos'][2]:params['crop_pos'][3],
                                    params['crop_pos'][0]:params['crop_pos'][1],
                                    ...]
        if params['flip']:
            numpyarray = numpyarray[:,
                                    ::-1,
                                    ...]

        if option==1:
            if colorjitter and params['colorjitter'] and random.random() > 0.1:
                # numpyarray += np.random.rand() * 30 - 15
                # numpyarray[numpyarray > 255] = 255
                # numpyarray[numpyarray < 0] = 0
                hsv = cv2.cvtColor(numpyarray, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] += np.random.rand() * 70 - 35
                hsv[:, :, 1] += np.random.rand() * 0.3 - 0.15
                hsv[:, :, 2] += np.random.rand() * 50 - 25
                hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 360.)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1.)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255.)
                numpyarray = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # print numpyarray.shape

    if option == 1:
        if not normalize:
            numpyarray = numpyarray - np.asarray([122.675,116.669,104.008])
            numpyarray = numpyarray.transpose((2, 0, 1))[::-1,:,:].astype(np.float32)
        else:
            numpyarray = numpyarray.transpose((2, 0, 1)).astype(np.float32)/255.

    if option == 2:
        if not normalize:
            numpyarray = numpyarray - np.asarray([132.431, 94.076, 118.477])
            numpyarray = numpyarray.transpose((2, 0, 1))[::-1,:,:].astype(np.float32)
        else:
            numpyarray = numpyarray.transpose((2, 0, 1)).astype(np.float32)/255.

    if len(numpyarray.shape) == 3:
        torchtensor = torch.from_numpy(numpyarray.copy()).float()#.div(255)
    else:
        torchtensor = torch.from_numpy(np.expand_dims(numpyarray,axis=0).copy())

    if normalize:
        # torchtensor = torchtensor.div(255)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        torchtensor = normalize(torchtensor)

    return torchtensor


