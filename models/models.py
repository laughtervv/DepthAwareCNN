import torch

def create_model(opt, dataset=None):

    if opt.model == 'DeeplabVGG':
        from .Deeplab import Deeplab_Solver
        model = Deeplab_Solver(opt, dataset)
    elif opt.model == 'DeeplabVGG_HHA':
        from .Deeplab_HHA import Deeplab_HHA_Solver
        model = Deeplab_HHA_Solver(opt, dataset)
    elif opt.model == 'DeeplabResnet':
        from .Deeplab import Deeplab_Solver
        model = Deeplab_Solver(opt, dataset,'Resnet')

    print("model [%s] was created" % (model.name()))

    return model
