from torchvision import transforms
import torch
from argparse import ArgumentParser
from ipdb import set_trace
import numpy as np

class TransformsComposer():
    """
    Compose and setup the transforms depending command line arguments.
    Define a series of transforms, each transform takes a dictionnary
    containing a subset of keys from ['Flow', 'Image', 'GtMask'] and
    has to return the same dictionnary with content elements transformed.
    """
    def __init__(self, flow_normalisation, flow_augmentation) :
        transfs = []
        transfs.append(TrAugmentFlow(flow_augmentation))
        if flow_normalisation != 'none' :
            transfs.append(TrNormaliseFlow(flow_normalisation))
        self.TrCompose = transforms.Compose(transfs)

    def __call__(self, ret) :
        return self.TrCompose(ret)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TrNormaliseFlow.add_specific_args(parser)
        parser = TrAugmentFlow.add_specific_args(parser)
        return parser


class TrNormaliseFlow() :
    """
    Normalise input Flow

    Args :
        Name normalisation ( str ) : normalisation to use {median, max, mean}
    """

    def __init__(self, flow_normalisation) :
        self.flow_normalisation = flow_normalisation
        print(self.declare())

    def declare(self) :
        return f'Flow Normalisation : {self.flow_normalisation}'

    def __call__(self, ret) :
        """
        Args :
            ret : dictionnary containing at least "Flow"
        Return :
            ret dictionnary with Flow transform for one frame
                'Flow' : (2, W, H)
        """
        eps = torch.finfo(ret['Flow'].dtype).eps
        norms = torch.sqrt((ret['Flow']**2).sum(axis=0))

        if self.flow_normalisation == 'median' :
            ret['Flow'] /= torch.median(norms)
        elif self.flow_normalisation == 'mean' :
            ret['Flow'] /= torch.mean(norms)
        elif self.flow_normalisation == 'max' :
            ret['Flow'] /= torch.max(norms)
        elif self.flow_normalisation == 'znorm' :
            exp = ret['Flow'].mean(axis=(1,2), keepdim=True)
            var = ret['Flow'].var(axis=(1,2), keepdim=True)
            ret['Flow'] = (ret['Flow'] - exp)/np.sqrt(var+eps)
        elif self.flow_normalisation == 'none' :
            pass
        else :
            raise Exception(f'Flow normalisation {self.flow_normalisation} not implemented')
        return ret

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--flow_normalisation', help='Normalisation method to use for the flow map',
                            type=str, default='none', choices=['median', 'mean', 'max', 'znorm', 'none'])
        return parser


class TrAugmentFlow() :
    """
    Data augmentation techniques for optical flow fields

    Args :
        Name flow_augmentation (list str) : data augmentation to return
    """
    def __init__(self, flow_augmentation) :
        self.augs = []
        if 'globalmotion' in flow_augmentation :
            if 'globalmotionlight' in flow_augmentation : # Affine motion
                    self.augs.append(lambda x : self.globalmotion(x, alpha=2))
            if 'globalmotionfullquad' in flow_augmentation : # Quadratic motion
                    self.augs.append(lambda x : self.globalmotionfullquad(x, alpha=2))
        self.declare()

    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        """
        for aug in self.augs :
            ret = aug(ret)
        return ret

    def declare(self):
         print(f'Flow Transformations : {[aug for aug in self.augs]}')


    @staticmethod
    def globalmotion(ret, alpha) :
        """
        Add a global motion to the flow field
        Args :
          ret : dictionnary containing at least "Flow"
          alpha : magnitude of the global motion compared to the original flow
        Return :
          ret dictionnary with Flow transform for one frame
              'Flow' : (2, W, H)
        """
        if torch.rand(1).item() < 0.6 : # Affine Model
            params = (torch.rand(3, 2) - 0.5)*2
        else : # Rotation
            a = (torch.rand(1)- 0.5)*2
            params = torch.tensor([[0, a], [a, 0], [0, 0]])
        _, I , J = ret['Flow'].shape
        i, j = np.meshgrid(np.arange(I), np.arange(J), indexing='ij')
        i = ((i.flatten()/I) - 0.5)*2 # Scaling to have values between -1 and 1
        j = ((j.flatten()/J) - 0.5)*2 # Scaling between -1 and 1
        X = torch.tensor(np.stack([i, j, np.ones_like(i)]), dtype=torch.float32).T
        param_flo = (X@params).reshape(I,J,2).permute(2,0,1)
        mf = torch.sqrt((ret['Flow']  ** 2).sum(axis=0)).mean()
        mp = torch.sqrt((param_flo  ** 2).sum(axis=0)).mean()

        ret['Flow'] += param_flo * torch.rand(1) * alpha * (mf / mp)
        return ret

    @staticmethod
    def globalmotionfullquad(ret, alpha) :
        """
        Add a full quadratic global motion to the flow field
        Args :
          ret : dictionnary containing at least "Flow"
          alpha : magnitude of the global motion compared to the original flow
        Return :
          ret dictionnary with Flow transform for one frame
              'Flow' : (2, W, H)
        """
        params = (torch.rand(6, 2) - 0.5)*2
        _, I , J = ret['Flow'].shape
        i, j = np.meshgrid(np.arange(I), np.arange(J), indexing='ij')
        i = ((i.flatten()/I) - 0.5)*2 # Scaling to have values between -1 and 1
        j = ((j.flatten()/J) - 0.5)*2 # Scaling between -1 and 1
        X = torch.tensor(np.stack([i*j, i**2, j**2, i, j, np.ones_like(i)]), dtype=torch.float32).T
        param_flo = (X@params).reshape(I,J,2).permute(2,0,1)
        mf = torch.sqrt((ret['Flow']  ** 2).sum(axis=0)).mean()
        mp = torch.sqrt((param_flo  ** 2).sum(axis=0)).mean()

        ret['Flow'] += param_flo * torch.rand(1) * alpha * (mf / mp)
        return ret


    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--flow_augmentation', type=str, default='none')
        return parser
