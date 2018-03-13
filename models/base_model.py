import os
import numpy as np
from utils import util
import torch

def load_pretrained_model(net, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    # print state_dict.keys()
    # print own_state.keys()
    for name, param in state_dict.items():
        if name in own_state:
            # print name, np.mean(param.numpy())
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if strict:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print('Ignoring Error: While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class BaseModel():

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.training = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.num_classes = opt.label_nc
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.tensorborad_dir = os.path.join(self.opt.checkpoints_dir, 'tensorboard', opt.dataset_mode)
        self.model_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'model')
        util.mkdirs([self.tensorborad_dir, self.model_dir])

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.model_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label,network_label)
        if not save_dir:
            save_dir = self.model_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                # print torch.load(save_path).keys()
                # print network.state_dict()['Scale.features.conv2_1_depthconvweight']
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    print('Pretrained network has excessive layers; Only loading layers that are used' )
                except:
                    print('Pretrained network has fewer layers; The following are not initialized:' )
                    # from sets import Set
                    # not_initialized = Set()
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v
                    not_initialized=[]
                    # print(pretrained_dict.keys())
                    # print(model_dict.keys())
                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized+=[k]#[k.split('.')[0]]
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  

    def update_learning_rate():
        pass


    def load_pretrained_network(self, network, pretraineddir, epoch_label,strict=True):
        save_filename = '%s.pth' % (epoch_label)
        save_path = os.path.join(pretraineddir, save_filename)
        load_dict = torch.load(save_path, map_location=lambda storage, loc: storage)
        # print (load_dict.values().size())
        load_pretrained_model(network,load_dict,strict)
