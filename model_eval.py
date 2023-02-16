from Models.CoherenceNets.MethodeB import MethodeB
from csvflowdatamodule.CsvDataset import CsvDataModule
from utils.Callbacks import ResultsLogger, SaveResultsFig

import sys, torch, os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--save_dir', type=str, help='directory to save the results in', required=True)
parser.add_argument('--draw_figs', action='store_true', help="Generate figure (.png)")
parser.add_argument('--save_npy', action='store_true', help="Generate probability prediction (.npy)")
parser.add_argument('--save_mask', action='store_true', help="Generate binary mask (.png)")
parser.add_argument('--steps', nargs='+', type=str, default=['test', 'val'])
parser.add_argument('--binary_method_gen', help='Method to use to produce binary masks', type=str,
                    choices=['fair', 'exceptbiggest'], default=None)
parser = CsvDataModule.add_specific_args(parser)
args = parser.parse_args()

Path(args.save_dir).mkdir(exist_ok=True)

#########################
##      Load Model     ##
#########################
model = MethodeB.load_from_checkpoint(args.ckpt, strict=False)
args.img_size = model.hparams.img_size
args.flow_normalisation = model.hparams.flow_normalisation
if args.binary_method_gen is not None :
    model.binary_method = args.binary_method_gen


#########################
##      Load Data     ##
#########################
args.data_path  = args.data_file+'_{}.csv'
if args.draw_figs :
    model.request.add('Image')
dm = CsvDataModule(request=model.request, **vars(args))
[dm.setup_dataset(step) for step in args.steps]

#########################
##      Run Eval       ##
#########################
for step in args.steps :
    print(f'Step : {step}')
    args.callbacks = []
    args.callbacks.append(ResultsLogger(filepath=os.path.join(args.save_dir, f'results_{step}.csv')))
    if args.draw_figs or args.save_npy or args.save_mask :
        args.callbacks.append(SaveResultsFig(save_dir=args.save_dir,
                              save_figure=args.draw_figs,
                              save_npy=args.save_npy,
                              save_mask=args.save_mask))

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.test(model, test_dataloaders=dm.get_dataloader(step))
