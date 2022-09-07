import pytorch_lightning as pl
import torch.nn as nn
from ipdb import set_trace
from argparse import ArgumentParser
import torch

class LitHead(pl.LightningModule):

    def __init__(self, **kwargs) :
        super().__init__()
        self.model = self.init_model(**kwargs)

    def init_model(self, features_backbone, nAffineMasks, **kwargs) :
        self.nAffineMasks = nAffineMasks
        return SingleMask(**kwargs, in_feats=features_backbone, out_feats=nAffineMasks)

    def forward(self, batch) :
        """
        Take the batch with "FeatureBackbone" : (b, L, W, H)
        and return the prediction (b, L, W, H)
        """
        return torch.softmax(self.model(batch['FeatureBackbone']), dim=1)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--nAffineMasks', '-L', type=int, default=2)
        return parser


class SingleMask(nn.Module) :
    def __init__(self, in_feats, out_feats, **kwargs) :
        super().__init__()
        self.pred_conv = nn.Conv2d(in_feats, out_feats, kernel_size=1)

    def forward(self, x) :
        return self.pred_conv(x)
