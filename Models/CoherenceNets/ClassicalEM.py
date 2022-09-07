from Models.CoherenceNets.MethodeB import MethodeB
import torch
from sklearn.linear_model import HuberRegressor
import numpy as np
from ipdb import set_trace
import cv2
from argparse import ArgumentParser


class ClassicalEM(MethodeB) :
    """
    Method performing predictions using a classical EM algo ( expectation )
    instead of training a network.
    """
    def __init__(self, **kwargs) :
        super().__init__(**kwargs) # Build Coherence Net

    def Expectation(self, theta, flow) :
        """
        Compute expected segmentation from theta
        Params :
            theta : parameters set for each layers and sample : (b, L, , J)
            flow (b, 2, I, J) : Flow Map
        Return :
            expected_segmentation (b, L, I, J) : Mask proba predictions
        """
        rsc = self.ComputeParametricFlow(theta, flow) #[ b, l, i*j, 2]
        b, L, I, J, _ = list(rsc.shape)
        flr = torch.repeat_interleave(flow.permute(0,2,3,1)[:,None], repeats = L ,dim=1 )# (b, L, I, J, 2 )
        errors_maps = self.vdist(rsc, flr) # (b, L, I, J)
        expected_segmentation = torch.softmax(-errors_maps, dim=1)  # (b, L, I, J)
        return expected_segmentation

    def prediction(self, batch) :
        """
        Produce a prediction for the batch using EM algo ( not trainable ).
        """
        b, K, I, J = batch['Flow'].shape
        L = self.hparams['nAffineMasks']
        ft = self.ft
        theta = (torch.rand((b, L, ft, K), device=batch['Flow'].device) - 0.5)*2
        #print(f'Init Theta :{theta}')

        for e in range(25) :
            batch['Pred'] = self.Expectation(theta, batch['Flow'])
            theta = self.ComputeTheta(batch)
        batch['Pred'].requires_grad_(True) # Avoid Error from Pytorch Lightning
        return batch
