import pytorch_lightning as pl
import flowiz
import torch
import wandb
from ipdb import set_trace
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ResultsLogger(pl.Callback) :
    def __init__(self, filepath, keys=['losses', 'jaccs', 'masks_usages']):
        super().__init__()
        self.fp = filepath
        self.keys = keys

    def setup(self, trainer, pl_module, stage) :
        print(f'Save results in {self.fp}')
        with open(self.fp, 'w') as f :
            f.write(f'epoch,step_label,file_name,'+','.join(self.keys)+'\n')

    @torch.no_grad()
    def write_results(self, imps, outputs, epoch, step_label) :
        with open(self.fp, 'a') as f :
            for i, imn in enumerate(imps) :
                f.write(f'{epoch},{step_label},{imn},'+','.join([f'{outputs[j][i].item():.3f}' for j in self.keys if j in outputs.keys()])+'\n')

    def batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, step_label):
        key_path = 'ImagePath' if 'ImagePath' in batch.keys() else 'FlowPath'
        self.write_results(batch[key_path], outputs, trainer.current_epoch, step_label)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx, 'test')

    def on_test_end(self, trainer, pl_module) :
        self.summary_path = self.fp.replace('.csv', '_summary.tsv')
        dfr = pd.read_csv(self.fp)
        dfr['sequence'] = dfr['file_name'].apply(lambda x : x.split('/')[-2])
        dfr['dataset'] = dfr['file_name'].apply(lambda x : x.split('/')[0])
        dfr.groupby('sequence').mean().to_csv(self.summary_path, sep='\t')
        self.summary_log = dfr.mean()
        self.summary_log.to_csv(self.summary_path, sep='\t', mode='a', header=False)
        print(f'Summary saved at : {self.summary_path}')

class SaveResultsFig(pl.Callback) :
    def __init__(self, save_dir, save_figure=True, save_mask=True, save_npy=False) :
        self.save_dir = save_dir
        self.save_figure = save_figure
        self.save_mask = save_mask
        self.save_npy = save_npy

        print(f'Saving Images in {self.save_dir}')

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch_out = outputs.pop('batch')
        key_path =  'ImagePath' if 'ImagePath' in batch_out.keys() else 'FlowPath'
        for i in range(len(outputs['losses'])):
            batch_out_idx = self.extract_dict(batch_out, i)
            outputs_idx = self.extract_dict(outputs, i)
            p = Path(f'{self.save_dir}/{batch_out[key_path][i]}')
            p.parent.mkdir(parents=True, exist_ok=True)

            if self.save_figure :
                # Results Image representation
                fig = pl_module.generate_result_fig(batch_out_idx, outputs_idx)
                plt.tight_layout()
                fig.savefig(p.with_suffix('.png'))
                plt.close(fig)

            if self.save_mask :
                im = Image.fromarray(batch_out_idx['PredMask'].numpy())
                im.save(str(p)+'_binary.png')
                # Save probability masp
            if self.save_npy :
                np.save(p.parent / (p.stem +'_proba.npy'), batch_out_idx['Pred'].numpy())

    @staticmethod
    def extract_dict(dict, idx):
        """
        Extract the index idx from dict an push to cpu if necessary
        """
        r_dict = {}
        for k in dict.keys() :
            if torch.is_tensor(dict[k]) :
                if dict[k].dim() > 0 :
                    r_dict[k] = dict[k][idx].cpu()
                else :
                    r_dict[k] = dict[k].cpu()
            else : r_dict[k] = dict[k]
        return r_dict
