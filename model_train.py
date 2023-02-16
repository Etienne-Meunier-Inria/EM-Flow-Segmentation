from Models.CoherenceNets.MethodeB import MethodeB
from csvflowdatamodule.CsvDataset import CsvDataModule
from utils.Callbacks import ResultsLogger

import sys, torch, os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pathlib import Path
from datetime import datetime

parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser = MethodeB.add_specific_args(parser)
parser = CsvDataModule.add_specific_args(parser)
args = parser.parse_args()

pl.seed_everything(123)

#########################
##      Load Model     ##
#########################
model = MethodeB(**vars(args))


#########################
##      Load Data      ##
#########################
args.data_path  = args.data_file+'_{}.csv'
dm = CsvDataModule(request=model.request, **vars(args))


#########################
##   Model Checkpoint  ##
#########################
if args.path_save_model :
    path = Path(args.path_save_model)
    path.mkdir(exist_ok=True)
    # We save the model with the lowest validation loss.
    args.callbacks = [pl.callbacks.ModelCheckpoint(args.path_save_model,
                                                   monitor='epoch_val_loss',
                                                   filename='{epoch}-{epoch_val_loss:.5f}',
                                                   mode='min',
                                                   save_top_k=1),
                      ResultsLogger(args.path_save_model+'/results.csv')]


#########################
##        Logger       ##
#########################

# Configure the pytorch lightning logger of your choice here.
logger = pl.loggers.CSVLogger(args.path_save_model)
logger.log_hyperparams(args)

#########################
##      Run Training   ##
#########################
trainer = pl.Trainer.from_argparse_args(args, logger=logger)
trainer.fit(model, dm)
