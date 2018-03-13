import torch.utils.data
from .base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'nyuv2':
        # from data.nyuv2_dataset import NYUDataset
        from data.nyuv2_dataset_crop import NYUDataset,NYUDataset_val
        dataset = NYUDataset()
        if opt.vallist!='':
            dataset_val = NYUDataset_val()
        else:
            dataset_val = None
    elif opt.dataset_mode == 'voc':
        from data.VOC_dataset import VOCDataset,VOCDataset_val
        dataset = VOCDataset()
        if opt.vallist!='':
            dataset_val = VOCDataset_val()
        else:
            dataset_val = None

    elif opt.dataset_mode == 'sunrgbd':
        from data.sunrgbd_dataset import SUNRGBDDataset,SUNRGBDDataset_val
        dataset = SUNRGBDDataset()
        if opt.vallist!='':
            dataset_val = SUNRGBDDataset_val()
        else:
            dataset_val = None

    elif opt.dataset_mode == 'stanfordindoor':
        from data.stanfordindoor_dataset import StanfordIndoorDataset, StanfordIndoorDataset_val
        dataset = StanfordIndoorDataset()
        if opt.vallist!='':
            dataset_val = StanfordIndoorDataset_val()
        else:
            dataset_val = None

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    if dataset_val != None:
        dataset_val.initialize(opt)
    return dataset,dataset_val

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset, self.dataset_val = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        if self.dataset_val != None:
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.nThreads))
        else:
            self.dataloader_val = None


    def load_data(self):
        return self.dataloader, self.dataloader_val

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
