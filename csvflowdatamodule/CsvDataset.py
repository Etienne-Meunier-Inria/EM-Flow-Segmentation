from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from argparse import ArgumentParser
import flowiz
from PIL import Image
import numpy as np
from ipdb import set_trace
import pytorch_lightning as pl
import os
from pathlib import Path
from torchvision import transforms, io
from torch.utils.data.dataloader import default_collate
from .Transforms import TransformsComposer
import torchvision

class FilesLoaders() :
    def __init__(self) :
        self.loaders = {
                        'Flow' :  self.load_flow,
                        'Image' :  self.load_image,
                        'GtMask' :  self.load_mask,
                        'FlowRGB' : self.load_image
                       }

    def load_file(self, path, type, img_size=None) :
        file = self.loaders[type](path)
        if img_size is not None:
            if not file.shape[-2:] == img_size:
                resize = transforms.Resize(img_size)
                file = resize(file[None])[0] # You need to add and remove a leading dimension because resize doesn't accept (W,H) inputs
        return file

    @staticmethod
    def load_flow(flow_path) :
        flow = torch.tensor(flowiz.read_flow(flow_path)).permute(2, 0, 1) # 2, i, j
        assert flow.ndim == 3, f'Wrong Number of dimension : {image_path}'
        return flow

    @staticmethod
    def load_image(image_path) :
        im = (torch.tensor(np.array(Image.open(image_path))).permute(2,0,1)/255.) - 0.5 # c, i, j : [-0.5; 0.5]
        assert im.ndim == 3, f'Wrong Number of dimension : {image_path}'
        return im

    @staticmethod
    def load_mask(image_path) :
        im = torch.tensor(np.array(Image.open(image_path))/255., dtype=torch.long) # i, j, c
        if im.ndim == 3 :
            im = im[:,:,0]
        assert im.ndim == 2, f'Wrong Number of dimension : {image_path}'
        return im



class CsvDataset(Dataset) :
    def __init__(self,
        data_path: str,
        base_dir: str,
        img_size: tuple, # i, j
        request: list, # List of fields you want in the folder
        subsample = 1,# percemtage of the dataset available ( if this is under 1 we subsample randomly )
        transform=None):
     super().__init__()

     self.img_size = img_size
     self.transform = transform
     self.base_dir = base_dir
     self.files = pd.read_csv(data_path)
     if subsample < 1 :
         self.files = self.files.sample(frac=subsample, random_state=123)
     print('request : ', request)
     assert set(request).issubset(self.files.columns), f'CSV is missing some requested columns columns : {list(self.files.columns)} Request : {request}'
     self.available_request = self.files.columns
     self.request = request.copy()
     if 'GtMask' in self.available_request : self.request.add('GtMask')
     self.fl = FilesLoaders()

    def __len__(self) :
        return len(self.files)

    def __getitem__(self, idx):
        ret = {}
        loc = self.files.iloc[idx]
        for type in self.request :
            try :
                if isinstance(loc[type], (np.integer, np.float)) :
                    assert loc[type] is not np.nan, f'Error in the Datasplit in {type}'
                    ret[type] = loc[type]
                elif isinstance(loc[type], (str)) :
                    ret[f'{type}Path'] =  loc[type]
                    ret[type] = self.fl.load_file(os.path.join(self.base_dir, loc[type]), type, self.img_size)
                else :
                    raise Exception(f'Data type of {loc[type]} not handled')
            except Exception as e :
                #print(e) # File does not exist
                return None
        return self.transform(ret)


class CsvDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str,
                       base_dir: str,
                       batch_size: int,
                       request: list, # List of fields you want in the folder
                       img_size : tuple,
                       subsample_train=1, # percentage of the train data to use for training.
                       shuffle_fit=True,
                       **kwargs) :
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.base_dir = base_dir
        self.request = request
        self.subsample_train = subsample_train
        self.shuffle_fit = shuffle_fit
        self.kwargs_dataloader = {'batch_size':self.batch_size,
                                  'collate_fn' : self.collate_fn,
                                  'num_workers': 8,
                                  'drop_last': False}
        self.set_transformations(**kwargs)

    def set_transformations(self, flow_normalisation, flow_augmentation, val_augment=False, **kwargs) :
        self.transforms = {}
        self.transforms['train'] = TransformsComposer(flow_normalisation, flow_augmentation)
        if val_augment :
            print('Enabling Data Augmentation on validation set')
            self.transforms['val'] = TransformsComposer(flow_normalisation, flow_augmentation)
        else :
            self.transforms['val'] = TransformsComposer(flow_normalisation, [])
        self.transforms['test'] = TransformsComposer(flow_normalisation, [])

    def setup(self, stage=None):
        print(f'Loading data in : {self.data_path} ------ Stage : {stage}')
        if stage == 'fit' or stage is None:
             self.dtrain = CsvDataset(self.data_path.format('train'), self.base_dir, self.img_size, self.request,
                                      subsample=self.subsample_train, transform=self.transforms['train'])
             self.dval = CsvDataset(self.data_path.format('val'), self.base_dir, self.img_size, self.request,
                                    transform=self.transforms['val'])
        if stage == 'test' or stage is None: # For now the val and test are the same
             self.dtest = CsvDataset(self.data_path.format('test'), self.base_dir, self.img_size, self.request,
                                     transform=self.transforms['test'])
        self.size(stage)

    def setup_dataset(self, step) :
        assert step in ['train', 'val', 'test'], f'Step {step} non valid'
        print(f'Loading data in : {self.data_path} ------ Stage : {step}')
        setattr(self, f"d{step}", CsvDataset(self.data_path.format(step), self.base_dir, self.img_size, self.request, transform=self.transforms[step]))

    def size(self, stage=None) :
        print('Size of dataset :')
        if stage == 'fit' or stage is None:
            print(f'\tTrain : {self.dtrain.__len__()} \t Val : {self.dval.__len__()}')
        if stage == 'test' or stage is None:
            print(f'\t Test : {self.dtest.__len__()}')

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    def train_dataloader(self):
        return DataLoader(self.dtrain, **self.kwargs_dataloader ,shuffle=self.shuffle_fit)

    def val_dataloader(self):
        return DataLoader(self.dval, **self.kwargs_dataloader, shuffle=self.shuffle_fit)

    def test_dataloader(self):
        return DataLoader(self.dtest, **self.kwargs_dataloader, shuffle=False)

    def get_sample(self, set=None) :
        if set == "train"  : return next(iter(self.train_dataloader()))
        elif set == "val"  : return next(iter(self.val_dataloader()))
        elif set == "test" : return next(iter(self.test_dataloader()))

    def get_dataloader(self, set=None) :
        if set == "train"  : return self.train_dataloader()
        elif set == "val"  : return self.val_dataloader()
        elif set == "test" : return self.test_dataloader()

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TransformsComposer.add_specific_args(parser)
        parser.add_argument('--batch_size', default=10, type=int)
        parser.add_argument('--subsample_train', default=1, type=float)
        parser.add_argument('--img_size', nargs='+', type=int, default=[128, 224])
        parser.add_argument('--base_dir', type=str, required=True)
        parser.add_argument('--data_file', type=str, required=True)
        parser.add_argument('--val_augment', action='store_true', help='Enable data Augmentation in validation')
        return parser
