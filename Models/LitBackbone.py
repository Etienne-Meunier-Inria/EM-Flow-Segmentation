import pytorch_lightning as pl
from ipdb import set_trace
from Models.Backbones.unet import UNet
from Models.Backbones.FakeModel import FakeModel
from argparse import ArgumentParser
import torch
from torchvision import models
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from ipdb import set_trace


class LitBackbone(pl.LightningModule):

    def __init__(self, inputs, **kwargs) :
        super().__init__()
        self.batch_prep = BatchPreparator(inputs)
        self.model = self.init_model(**kwargs)

    def init_model(self, backbone, features_backbone, **kwargs) :
        """Initialise the backbone model.

        Parameters
        ----------
        backbone : str
            Name of the backbone to use.
        features_backbone : int
            Number of channel in the output of the backbone.
        **kwargs : dict
            Dictionnary with extra arguments for the backbone.
        """
        if backbone == 'unet' :
            return UNet(num_classes = features_backbone,
                        input_channels=self.batch_prep.in_batch,
                        num_layers=kwargs['unet.num_layers'],
                        features_start=kwargs['unet.features_start'], **kwargs)
        elif backbone == 'fake':
            return FakeModel(features_backbone,
                             input_channels=self.batch_prep.in_batch)
        else :
            print(f'Backbone {backbone} not available')

    def forward(self, batch) :
        out, batch['hidden_features'] = self.model(self.batch_prep.prepare_batch(batch))
        return out

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = UNet.add_specific_args(parser)
        parser.add_argument('--features_backbone', type=int, default=32)
        parser.add_argument('--backbone', '-bb', type=str, choices=['unet', 'fake'], default='unet')
        parser.add_argument('--inputs', nargs='+', type=str, default=['Flow'])
        return parser


class BatchPreparator() :
    TYPES = { 'Flow' : 2,
              'Image' : 3,
              'FlowRGB' : 3}
    def __init__(self, inputs) :
        """
        The batch preparator set up the batch before giving it to the model
        It can concatenate the different elements from the batch
        inputs list(str) : list of the names of the input to concatenate in the order
        """
        self.inputs = inputs
        self.in_batch = self.compute_len(inputs)

    def prepare_batch(self, batch) :
        """
        prepare_batch combining inputs requested from the batch.
        """
        return torch.cat([batch[k] for k in self.inputs], axis=1)

    def get_lengths(self) :
        """
        Compute and return the list of all channel sizes
        """
        return [BatchPreparator.TYPES[k] for k in self.inputs]

    @staticmethod
    def compute_len(inputs) :
        """
        Compute and return the total number of channels
        """
        return sum([BatchPreparator.TYPES[k] for k in inputs])
