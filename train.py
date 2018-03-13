import time
from tensorboardX import SummaryWriter
from collections import OrderedDict
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import utils.util as util
from utils.visualizer import Visualizer
import os
import numpy as np
import torch
from torch.autograd import Variable
import time

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
ioupath_path = os.path.join(opt.checkpoints_dir, opt.name, 'MIoU.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
        
    try:
        best_iou = np.loadtxt(ioupath_path, dtype=float)
    except:
        best_iou = 0.
    print('Resuming from epoch %d at iteration %d, previous best IoU %f' % (start_epoch, epoch_iter, best_iou))
else:
    start_epoch, epoch_iter = 1, 0
    best_iou = 0.

data_loader = CreateDataLoader(opt)
dataset, dataset_val = data_loader.load_data()
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

model = create_model(opt, dataset.dataset)
# print (model)
visualizer = Visualizer(opt)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
for epoch in range(start_epoch, opt.nepochs):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    model.model.train()
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        ############## Forward and Backward Pass ######################
        model.forward(data)
        model.backward(total_steps, opt.nepochs * dataset.__len__() * opt.batchSize + 1)

        ############## update tensorboard and web images ######################
        if total_steps % opt.display_freq == 0:
            visuals = model.get_visuals(total_steps)
            visualizer.display_current_results(visuals, epoch, total_steps)

        ############## Save latest Model   ######################
        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
        # print time.time()-iter_start_time

    # end of epoch
    model.model.eval()
    if dataset_val!=None:
        label_trues, label_preds = [], []
        for i, data in enumerate(dataset_val):
            seggt, segpred = model.forward(data,False)
            seggt = seggt.data.cpu().numpy()
            segpred = segpred.data.cpu().numpy()

            label_trues.append(seggt)
            label_preds.append(segpred)

        metrics = util.label_accuracy_score(
            label_trues, label_preds, n_class=opt.label_nc)
        metrics = np.array(metrics)
        metrics *= 100
        print('''\
                Validation:
                Accuracy: {0}
                Accuracy Class: {1}
                Mean IU: {2}
                FWAV Accuracy: {3}'''.format(*metrics))
        model.update_tensorboard(metrics,total_steps)
    iter_end_time = time.time()

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch+1, opt.nepochs, time.time() - epoch_start_time))
    if metrics[2]>best_iou:
        best_iou = metrics[2]
        print('saving the model at the end of epoch %d, iters %d, loss %f' % (epoch, total_steps, model.trainingavgloss))
        model.save('best')

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d, loss %f' % (epoch, total_steps, model.trainingavgloss))
        model.save('latest')
        model.save(epoch)
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')

