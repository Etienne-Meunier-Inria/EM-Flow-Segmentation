import torch
from DataLoadingModule import DataLoadingModule
from ScoreModule import ScoreModule
import os, sys
import matplotlib.pyplot as plt
sys.path.append('..')
from ipdb import set_trace
import numpy as np

class BinarisationModule() :
    def __init__(self, binary_method) :
        """BinarisationModule handles turning a segmentation probability map
        with potentially K classes into a binary map with foreground (1) and
        background (0).

        Parameters
        ----------
        binary_method (str) : name of the method to do binarisation
        """
        self.request = set()
        self.select_binary_method(binary_method)


    def select_binary_method(self, binary_method):
        """Select the binary method function to use depending on the name.

        Parameters
        ----------
        binary_method (str) : name of the method to do binarisation
        """

        if binary_method == 'exceptbiggest' :
            self.binary_method = self.exceptbiggest
        elif binary_method == 'fair' :
            self.request.add('GtMask')
            self.binary_method = self.fair
        else :
            raise Exception(f'Binary method : {binary_method} not implemented')


    def binarise(self, d) :
        """Perform a binarisation of the probability module based on the method.

        Parameters
        ----------
        d (dict) : dictionnary containing tensors for bianrisation, potentially :
            'Pred' (torch.tensor - float): Probability segmentation map (b, l, i, j) with l classes
            'GtMask' (torch.tensor - bool) : Ground Truth binary segmentation map (b, i, j)

        Returns
        -------
        None but add to dict :
            'PredMask' (torch.tensor) : Binary segmentation mask ( b, i, j) with frgd : 1 and bkgd : 0
        """
        assert d['Pred'].ndim ==4, f'Pred input have {d["Pred"].ndim} dimension when 4 are required'
        assert d['Pred'].sum(axis=1).allclose(torch.tensor(1, dtype=d["Pred"].dtype)), 'Pred should sum to 1 at every position'
        if 'GtMask' in d.keys() :
            d['GtMask'] = self.binarise_gt(d['GtMask'])
        d["PredMask"] = self.binary_method(**d)
        assert [d['Pred'].shape[i] for i in [0,2,3]] == list(d['PredMask'] .shape), f'Error in shape : Pred ({d["Pred"].shape}) Binary Mask ({d["PredMask"].shape})'
        assert d['PredMask'].dtype == torch.bool, f'Binary Mask should be bool currently is {d["PredMask"].dtype}'

    @staticmethod
    def binarise_gt(gt) :
        """Binarise GtMask, all values above 0 are turn to 1.

        Parameters
        ----------
        gt (torch.tensor - int64): GtMask(b, i, j) with potentially several distinct value.

        Returns
        -------
        type
            gt (torch.tensor - bool) : GtMask (b, i, j) with boolean variables
        """
        return gt.to(bool)


    @staticmethod
    def exceptbiggest(Pred, **k) :
        """
        Select all segments except the biggest one.

        Parameters
        ----------
        Pred (torch.tensor): Probability segmentation map (b, l, i, j) with l classes


        Returns
        -------
        binary_mask (torch.tensor) : Binary segmentation mask ( b, i, j) with frgd : 1 and bkgd : 0
        """
        b, L, I, J = Pred.shape
        argmax = Pred.argmax(1)
        idxmax = argmax.flatten(1).mode().values
        binary_mask = (argmax != idxmax[:,None, None].repeat(1,argmax.shape[1], argmax.shape[2]))
        return binary_mask


    @staticmethod
    def fair(Pred, GtMask, **k) :
        """Segment masks using argmax(p)
           Select segments that overlap with foreground for more than half their pixels
           NB : Not valid for benchmark as it uses the GT

        Parameters
        ----------
        Pred (torch.tensor): Probability segmentation map (b, l, i, j) with l classes
        GtMask (torch.tensor) : Binary ground truth mask


        Returns
        -------
        binary_mask (torch.tensor) : Binary segmentation mask ( b, i, j) with frgd : 1 and bkgd : 0
        """
        b, L, I, J = Pred.shape
        argmax = Pred.argmax(1, keepdims=True)
        binmax = torch.zeros_like(Pred)
        binmax.scatter_(1, argmax, 1)
        spgtmask = GtMask[:,None].repeat_interleave(L, dim=1)
        si = ((binmax *spgtmask).sum(axis=(2,3)) / binmax.sum(axis=(2,3))) > 0.5
        binary_mask = (binmax * si[:,:,None,None]).sum(axis=1).to(bool)
        return binary_mask



if __name__=='__main__' :
    data_file = 'DAVIS_D16Split_val'
    data_base_dir = os.environ['Dataria']
    pred_base_dir = f'{os.environ["Dataria"]}/Models/SegGrOptFlow/vir/m4mmn3jp/'
    bnm = BinarisationModule('exceptbiggest')
    dlm = DataLoadingModule(data_file, data_base_dir, pred_base_dir, bnm.request)
    d = dlm.__getitem__(4)
    bnm.binarise(d)
