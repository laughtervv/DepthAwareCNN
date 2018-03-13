from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os

####evaluation metrics

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class, returniu = False):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    if returniu:
        return acc, acc_cls, mean_iu, fwavacc, iu[freq > 0]
    else:
        return acc, acc_cls, mean_iu, fwavacc

###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

class Colorize(object):
    def __init__(self, n=35):
        n = 256
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, inputmode=''):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, inputmode))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if inputmode=='div255-mean':
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif inputmode=='divstd-mean':
        i_max = np.max(image_numpy)
        i_min = np.min(image_numpy)
        image_numpy = np.transpose((image_numpy-i_min)/(i_max-i_min) * 255.0, (1, 2, 0))
    elif inputmode=='bgr-mean':
        image_numpy = np.transpose(image_numpy, (1, 2, 0))[:,:,::-1] + np.asarray([122.675,116.669,104.008])
        # print(image_numpy.max(),image_numpy.min())
    else:
        # print('depth')
        image_numpy = np.transpose(image_numpy, (1, 2, 0))[:,:,::-1] * 255.0
    # image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

# Converts a one-hot tensor into a colorful label map
colormap = Colorize(255)
def tensor2label(label_tensor, n_label, imtype=np.uint8, colorize = True):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()    
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    if colorize:
        label_tensor = colormap(label_tensor)
        label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    else:
        label_numpy = np.squeeze(label_tensor.numpy())
        # print (np.unique(label_numpy.astype(imtype)))
    return label_numpy.astype(imtype)

def save_image(image_numpy, image_path, imagesize = None):
    image_pil = Image.fromarray(image_numpy)
    if imagesize is not None:
        img_w, img_h = imagesize
        image_pil = image_pil.resize((img_w, img_h), Image.NEAREST)
    # print(np.unique(np.asarray(image_pil)))
    image_pil.save(image_path)
    # if len(image_numpy.shape)==2:
    #     image_pil = Image.open(image_path)
    #     print(image_path,np.unique(np.asarray(image_pil)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
