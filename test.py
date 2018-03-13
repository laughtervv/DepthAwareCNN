import os
import numpy as np
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import utils.util as util
from utils.visualizer import Visualizer
from utils import html
from torch.autograd import Variable

opt = TestOptions().parse(save=False)
opt.nThreads = 1   
opt.batchSize = 1  
opt.serial_batches = True  # no shuffle

data_loader = CreateDataLoader(opt)
dataset, _ = data_loader.load_data()
model = create_model(opt,data_loader.dataset)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, '%s: %s' % (opt.name, pt.which_epoch))
# test


label_trues, label_preds = [], []

model.model.eval()
tic = time.time()

accs=[]
for i, data in enumerate(dataset):
    if i >= opt.how_many and opt.how_many!=0:
        break
    seggt, segpred = model.forward(data,False)
    print time.time() - tic
    tic = time.time()

    seggt = seggt.data.cpu().numpy()
    segpred = segpred.data.cpu().numpy()

    label_trues.append(seggt)
    label_preds.append(segpred)

    visuals = model.get_visuals(i)
    img_path = data['imgpath']
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

metrics0 = util.label_accuracy_score(
    label_trues, label_preds, n_class=opt.label_nc, returniu=True)
metrics = np.array(metrics0[:4])
metrics *= 100
print('''\
        Accuracy: {0}
        Accuracy Class: {1}
        Mean IU: {2}
        FWAV Accuracy: {3}'''.format(*metrics))

webpage.save()
