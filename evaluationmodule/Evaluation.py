import torch, os, sys
from DataLoadingModule import DataLoadingModule
from BinarisationModule import BinarisationModule
from ScoreModule import ScoreModule
from SaveModule import SaveModule
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--scoring_method', '-sm', type=str, choices=['db_eval_iou', 'bbox_jacc'], default=None)
parser.add_argument('--binary_method', '-bm', type=str, choices=['exceptbiggest', 'fair'], default='exceptbiggest')
parser.add_argument('--model_id', '-mi', type=str)
parser.add_argument('--model_base_dir', type=str)
parser.add_argument('--data_base_dir', type=str)
parser.add_argument('--data_file', '-df', type=str)
args = parser.parse_args()

if args.scoring_method == None :
    if 'Moca' in args.data_file :
        args.scoring_method = 'bbox_jacc'
    else :
        args.scoring_method = 'db_eval_iou'

hparams = {}

hparams['data_file'] = args.data_file
hparams['data_base_dir'] = os.environ['Dataria']
hparams['model_id'] = args.model_id
hparams['pred_base_dir'] = f'{args.model_base_dir}/{hparams["model_id"]}/'
hparams['scoring_method'] = args.scoring_method
hparams['binary_method'] = args.binary_method

scm = ScoreModule(hparams['scoring_method'])
bnm = BinarisationModule(hparams['binary_method'])

bnm.request.add('Flow')
request = scm.request | bnm.request
dlm = DataLoadingModule(**hparams, request=request)

dld = DataLoader(dlm, collate_fn=dlm.collate_fn, batch_size=1, num_workers=4) # Dataloader
svm = SaveModule(save_dir=hparams['pred_base_dir'])


for i, d in enumerate(tqdm(dld)) :
    try :
        bnm.binarise(d)
        scm.score(d)
        scm.stat_masks(d)
        svm.write_result(d)
        svm.save_binary(d)
        if i % 10 == 0 :
            svm.generate_fig(d)
    except Exception as e :
        print(e)
        pass
hparams.update(svm.summarise_csv(dlm.dst.files))
svm.save_config(hparams)
