import flowiz
from PIL import Image
import torch
import torch.nn as nn
from argparse import Namespace
import pytorch_lightning as pl
from argparse import ArgumentParser
from ipdb import set_trace
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from utils.evaluations import *
from ipdb import set_trace
from Models.LitBackbone import LitBackbone
from Models.LitHead import LitHead

class LitSegmentationModel(pl.LightningModule) :
    def __init__(self, **kwargs) :
        super().__init__()
        self.hparams.update(kwargs)
        self.backbone_model = LitBackbone(**kwargs)
        self.head_model = LitHead(**kwargs)
        self.d = Namespace(**{'L': self.head_model.nAffineMasks})
        self.request = set(self.backbone_model.batch_prep.inputs)

    def configure_optimizers(self):
        if self.hparams['optim.name'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['optim.lr'],
             weight_decay=self.hparams['optim.weight_decay'])
        elif self.hparams['optim.name'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams['optim.lr'],
             weight_decay=self.hparams['optim.weight_decay'])
        print(f'Optimizer : {optimizer}')
        return optimizer

    def prediction(self, batch) :
        """
        Produce a prediction for a segmentation using the model and the batch
        """
        batch['FeatureBackbone'] = self.backbone_model.forward(batch)
        batch['Pred'] = self.head_model.forward(batch)
        return batch

    def step(self, batch, step_label):
        batch = self.prediction(batch)
        assert  batch['Pred'].sum(axis=1).allclose(torch.tensor([1], device=batch['Pred'].device).to(torch.float)),\
               'Prediction must sum to 1 for the masks'
        evals = self.Criterion(batch)
        self.Evaluations(batch, evals) # Add evaluations to evals dict
        self.log_dict({f'{step_label}/{k}':v for k, v in evals.items() if v.dim() == 0})
        return evals, batch

    def training_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'train')
        return evals

    def validation_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'val')
        return evals

    def test_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'test')
        evals.update({"batch":dict([(k, v.cpu()) if torch.is_tensor(v) else (k, v) for k,v in batch.items()])}) # For displaying we include the batch in the output
        return evals

    def epoch_end(self, outputs, step_label) :
        '''
        General function called at the end of the epoch that will log epoch wise metrics
        '''
        epoch_avg_loss = sum([o['loss'] for o in outputs])/len(outputs)
        self.log(f'epoch_{step_label}_loss', epoch_avg_loss)

    def validation_epoch_end(self, validation_step_outputs):
        self.epoch_end(validation_step_outputs, 'val')

    def training_epoch_end(self, training_step_outputs):
        self.epoch_end(training_step_outputs, 'train')

    def Evaluations(self, batch, evals) :
        """
        Run evaluations of the binary masks and add the key to evals
        Args : batch with at least keys
                    'Pred' (b, L, I, J) : Mask proba predictions with sofmax over dim=1
                    'GtMask' binary (b, I, J) : ground truth foreground mask
               evals with a least keys
                    'losses' (b) : Loss per image
        """
        self.generate_binary_mask(batch) # Generate Binary Maks
        evals['loss'] = evals['losses'].mean() # Loss is necessary for backprop in Pytorch Lightning
        a = batch['Pred'].argmax(axis=1).flatten(1).detach()
        evals['masks_usages'] = torch.tensor([len(a[i].unique()) for i in range(a.shape[0])])
        if 'GtMask' in batch :
            evals['jaccs'] = db_eval_iou(batch['GtMask'], batch['PredMask'])
            evals['jacc'] = evals['jaccs'].mean()
        evals['entropy'] = evals['entropies'].mean()
        evals['coherence_loss'] = evals['coherence_losses'].mean()

    def generate_result_fig(self, batch, evals) :
        """
        Compute the results for the given batch using th model and generate
        a figure presenting the results
        batch : images and modalities
        evals : evaluation metrics dict
        """
        sh = lambda x : flowiz.convert_from_flow(x.detach().cpu().permute(1,2,0).numpy())
        with torch.no_grad() :
            fig, ax = plt.subplots(2, 3, figsize = (15,10))
            if 'Image' in batch.keys() :
                ax[0,0].set_title('Image')
                ax[0,0].imshow(batch['Image'].permute(1,2,0)+0.5) # By default the image is normalised between [-0.5;0.5]
            if 'Flow' in batch.keys() :
                ax[0,1].set_title(f'Flow min: {torch.min(batch["Flow"]):.2f} max: {torch.max(batch["Flow"]):.2f}')
                ax[0,1].imshow(sh(batch['Flow']))
            ax[1,0].set_title(f'Pred : {evals["losses"]:.3f}')
            ax[1,0].imshow(batch['Pred'].argmax(0)) # For BCE Net pred 1 is object
            ax[1,1].set_title('PredMask')
            ax[1,1].imshow(batch['PredMask'])
            if 'GtMask' in batch.keys() :
                ax[1,2].set_title(f'GtMask {evals["jaccs"]:.3f}')
                ax[1,2].imshow(batch['GtMask'])
            else :
                ax[1,2].set_title(f'Confidence (max proba)')
                ax[1,2].imshow(batch['Pred'].max(axis=0).values)

            ax = self.custom_figs(ax, batch, evals) # Add axes corresponding to the class
        return fig

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LitBackbone.add_specific_args(parser)
        parser = LitHead.add_specific_args(parser)
        parser.add_argument('--model_type', '-mt', type=str, choices=['coherence_B', 'classicalEM'], default='coherence_B')
        parser.add_argument('--path_save_model', type=str)
        parser.add_argument('--optim.name', type=str, choices=['Adam', 'RMSprop'], default='Adam')
        parser.add_argument('--optim.lr', type=float, default=1e-4)
        parser.add_argument('--optim.weight_decay', type=float, default=0)
        return parser
