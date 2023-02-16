import torch
import numpy as np
import sys
from Models.LitSegmentationModel import LitSegmentationModel
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import flowiz
import warnings
from utils.distance_metrics import VectorDistance

class CoherenceNet(LitSegmentationModel) :

    def __init__(self, img_size, binary_method, v_distance, entropy_weight, param_model, **kwargs) :
        """Coherence Network

        Parameters
        ----------
        img_size : (int, int)
            Size of the input optical flow
        binary_method : str
            Method to use to transform the multimask segmentation in foreground/background mask. For display and evaluation.
        v_distance : str
            Name of the function to use to compute distance between vectors.
        entropy_weight : float
            Ponderation of the entropy loss compared to the other losses.
        param_model : str
            Name of the parametric model to use to represent the flow ( Affine / Quadratic)
        """

        super().__init__(**kwargs) # Build Lit Segmentation Model
        self.request.add('Flow')
        self.init_crit(*img_size, param_model)
        self.entropy_weight = entropy_weight
        self.hparams.update({'img_size':img_size,
                             'binary_method': binary_method,
                             'coherence_loss': 'pieceFit',
                             'v_distance':v_distance,
                             'entropy_weight':entropy_weight,
                             'param_model':param_model})
        self.setup_binary_method(binary_method)
        self.vdist = VectorDistance(v_distance)

    def init_crit(self, I, J, param_model) :
        self.d.I = I
        self.d.J = J
        Xo = self.init_Xo(I, J, param_model)
        self.register_buffer('XoT', Xo.T) # [ i*j , 3]
        self.ft = Xo.shape[0]

    @staticmethod
    def init_Xo(I, J , param_model) :
        """Generate and return XoT for the given shape

        Parameters
        ----------
            I (int) : height
            J (int) : width
            param_model (str): Type of parametric model to generate (Affine /  Quadratic )

        Returns
        -------
            Xo : Features for regression (ft, I*J) depending on parametric model
        """
        i, j = np.meshgrid(np.arange(I), np.arange(J), indexing='ij')
        i = i.flatten()/I # Scaling to have values between 0 and 1
        j = j.flatten()/J
        if param_model == 'Affine' :
            return(CoherenceNet.init_affine(i, j))
        elif param_model == 'Quadratic' :
            return(CoherenceNet.init_quadratic(i, j))

    @staticmethod
    def init_affine(i, j) :
        print('Init Affine Motion Model')
        Xo = torch.tensor(np.stack([i, j, np.ones_like(i)]), dtype=torch.float32) # Removed T [3, i*j]
        return Xo

    @staticmethod
    def init_quadratic(i, j) :
        print('Init Quadratic Motion Model')
        Xo = torch.tensor(np.stack([i*j, i**2, j**2, i, j, np.ones_like(i)]), dtype=torch.float32) # Removed T [6, i*j]
        return Xo

    @staticmethod
    def ComputeParametricFlow(theta, flow, XoT) :
        """
        Compute a parametric flow by layer using previously computed parameters theta
        Params :
            theta : parameters set for each layers and sample : (b, L, I, J)
            flow (b, 2, I, J) : Original Flow Map ( before data augmentation )
            XoT : Features for regression (I*J, ft) depending on parametric model
        Returns :
            'rsc' (b, L, I, J, 2) : Parametric flow for each layer and each position
             Add to batch : reconstruction (b, 2, I, J) : Piecewise parametric flow as a composition of layers
        """
        b, K, I, J = flow.shape
        _, ft = XoT.shape
        _, L, _, _ = theta.shape
        X = XoT.unsqueeze(0).expand(b*L, -1, -1) # [b*l, i*j, 3]
        theta = theta.view(b*L, ft, K) # [b, l, ft, 2]

        rsc_flat = torch.bmm(X, theta) # [b*l, i*j, 2]
        rsc = rsc_flat.view((b, L, I, J, K)) #[ b, l, i, j, 2]
        return rsc

    @torch.no_grad() # Results of any computation in the function will not have grad
    def ComputeFlowReconstruction(self, theta, batch):
        """
        Compute flow reconstruction for display purpose.
        Params :
            theta : parameters set for each layers and sample : (b, L, I, J)
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions
                'Flow'( b, 2, I, J) : Original Flow Map ( before data augmentation )
        Return :
            reconstruction (b, 2, I, J) : Piecewise parametric flow as a composition of layers
        """
        rsc = self.ComputeParametricFlow(theta, batch['Flow'], self.XoT) #[ b, l, i*j, 2]
        reconstruction = (rsc * batch['Pred'].unsqueeze(-1)).sum(axis=1) # [b, I, J, 2]
        return  reconstruction.permute(0, 3, 1, 2) # [b, 2, i, j]

    def Criterion(self, batch) :
        """
        Compute the coherence loss of the masks / preds depending on the given flow
        Params :
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions
                'Flow'( b, 2, I, J) : Original Flow Map ( before data augmentation )
        Returns :
            evals dictionnary with :
                losses (b) : loss for each prediction of the batch ( coherence + entropy )
        """
        b, L, I, J = list(batch['Pred'].shape)

        # Optimzation in Theta
        theta = self.ComputeTheta(batch).detach()
        batch['theta'] = theta

        # Compute Loss for optimisation of the network
        coherence_losses = self.CoherenceLoss(theta, batch['Pred'], batch['Flow'], self.XoT, self.vdist)
        entropies = -(batch['Pred']*torch.log2(torch.clamp(batch['Pred'], min=1e-4))).sum(axis=(1,2,3)) / (I*J) # [b]
        losses = coherence_losses - self.entropy_weight*entropies

        # Reconstruction for plotting
        batch['ParametricFlow'] = self.ComputeFlowReconstruction(theta, batch)
        return {'losses' : losses,
                'coherence_losses':coherence_losses,
                'entropies':entropies} # [b]

    @staticmethod
    def CoherenceLoss(theta, pred, flow, XoT, vdist) :
        """
        Compute and return the coherence loss of the model given theta and the batch.
        Params :
            theta : parameters set for each layers and sample : (b, L, ft, 2)
            pred (b, L, I, J) : Mask proba predictions
            flow (b, 2, I, J) : Original Flow Map
            XoT (I*J, ft) : Features for regression depending on parametric model
            vdist : distance function to compute the loss
        Return  :
            coherence_losses (b): coherence loss for each sample of the batch
        """
        rsc = CoherenceNet.ComputeParametricFlow(theta, flow, XoT) #[ b, l, i*j, 2]
        b, L, I, J, _ = list(rsc.shape)
        flr = torch.repeat_interleave(flow.permute(0,2,3,1)[:,None], repeats = L ,dim=1 )# (b, L, I, J, 2 )
        coherence_losses = (pred * vdist(rsc, flr)).sum(axis=(1,2,3))/(I*J)
        return coherence_losses

    def setup_binary_method(self, binary_method) :
        """
        Add to the request necessary inputs for the requested binary methods
        """
        if binary_method in ['optimal', 'optimax'] : self.request.add('GtMask')
        self.binary_method = binary_method


    def generate_binary_mask(self, batch):
        """
        Takes as input a batch of propability masks and return a binary mask of the foreground pixels for evaluation
            - 'exceptbiggest' : choose all mask as foreground except the bigggest one.
            - 'fair' : segment masks using argmax(p), select segments that overlap with foreground for more than half their pixels
        Args :
            batch : dictionnary with at least 2 keys
                'Pred' (b, L, I, J) : Mask proba predictions with sofmax over dim=1
        Returns : None but add key 'PredMask' binary (b, I, J) to batch
        """
        b, L, I, J = batch['Pred'].shape
        if self.binary_method == 'exceptbiggest' :
            b, L, I, J = batch['Pred'].shape
            argmax = batch['Pred'].argmax(1)
            idxmax = argmax.flatten(1).mode().values
            batch['PredMask'] = (argmax != idxmax[:,None, None].repeat(1,argmax.shape[1], argmax.shape[2]))
        elif self.binary_method == 'fair' :
            argmax = batch['Pred'].argmax(1, keepdims=True)
            binmax = torch.zeros_like(batch['Pred'])
            binmax.scatter_(1, argmax, 1)
            spgtmask = batch['GtMask'][:,None].repeat_interleave(L, dim=1)
            si = ((binmax *spgtmask).sum(axis=(2,3)) / binmax.sum(axis=(2,3))) > 0.5
            batch['PredMask'] = (binmax * si[:,:,None,None]).sum(axis=1).to(bool)


    def custom_figs(self, ax, batch, evals) :
        """
        Generate figures depending on the model
        ax : list of axes to add the figure
        batch : images and modalities
        evals : evaluation metrics dict
        """
        sh = lambda x : flowiz.convert_from_flow(x.detach().cpu().permute(1,2,0).numpy())
        ax[0,2].set_title('ParametricFlow')
        ax[0,2].imshow(sh(batch['ParametricFlow']))
        return ax

    @staticmethod
    def add_specific_args(parent_parser):
        parent_parser = CoherenceNet.__bases__[0].add_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--binary_method', help='Method to use to produce binary masks', type=str,
                            choices=['fair', 'exceptbiggest'], default='exceptbiggest')
        parser.add_argument('--entropy_weight', help='Weight to balance the likelihood and entropy in loss',
                            type=float, default=0.01)
        parser.add_argument('--v_distance', help='Vector distance metric to use in the computation', type=str,
                            choices=['squared', 'l2', 'l1'],
                            default='l1')
        parser.add_argument('--param_model', help='Model to use to fit and compute the parametric flow', type=str,
                            choices=['Affine', 'Quadratic'], default='Quadratic')
        return parser
