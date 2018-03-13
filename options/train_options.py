### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')        
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument('--loadfroms', action='store_true', help='continue training: load from 32s or 16s')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--use_softmax', action='store_true', help='if specified use softmax loss, otherwise log-softmax')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--nepochs', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--iterSize', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--maxbatchsize', type=int, default=-1, help='# of iter at starting learning rate')
        self.parser.add_argument('--warmup_iters', type=int, default=500, help='# of iter at starting learning rate')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.00025, help='initial learning rate for adam')
        self.parser.add_argument('--lr_power', type=float, default=0.9, help='power of learning rate policy')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
        self.parser.add_argument('--wd', type=float, default=0.0004, help='weight decay for sgd')

        self.isTrain = True
