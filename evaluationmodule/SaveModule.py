from datetime import datetime
from pathlib import Path
import json
import math
import flowiz as fl
import matplotlib.pyplot as plt
from flowiz import flowiz
import pandas as pd
from ipdb import set_trace
from PIL import Image

class SaveModule() :
    def __init__(self, save_dir) :
        """Save Module extract results and config to a directory.

        Parameters
        ----------
        save_dir (str) : path where to save the evaluation directory
        hparams (dict) : dictionnary with all parameters to save.
        """
        self.create_directory(save_dir)
        self.init_csv()

    def summarise_csv(self, data_file) :
        self.csv.close()
        r = pd.read_csv(self.csv_path, usecols = lambda x : x!='StatMasks')
        r = r.drop_duplicates('ImagePath')
        m = pd.merge(data_file, r, left_on='Image', right_on='ImagePath', how='left')
        suma = {}
        suma['missing_rows'] = int(m['PredPath'].isnull().sum())
        suma['total_rows'] = int(len(m))
        suma['mean_score'] = float(r['Score'].mean())
        m.to_csv(self.path_save/f'results_{self.eval_name}.csv')
        return suma

    def create_directory(self, save_dir) :
        """Create Save directory and define a name for the evaluation based
           on the timestamp.

        Parameters
        ----------
        save_dir (str) : path where to save the evaluation directory
        """

        self.eval_name = 'eval_'+datetime.now().strftime("%m%d%y%H%M%S")
        self.path_save = Path(save_dir+'/'+self.eval_name)
        self.path_save.mkdir(exist_ok=True)

    def save_config(self, hparams) :
        """Save configuration to a json file.

        Parameters
        ----------
        hparams : dictionnary with all parameters to save.
        """
        json_path = self.path_save / f'{self.eval_name}.json'
        with open(json_path, 'w') as js :
            json.dump(hparams, js, indent=4, sort_keys=True)
        print(f'Wrote json configuration in {json_path}')

    def init_csv(self) :
        """Init csv file to save result for each image.
        """
        self.csv_path = self.path_save / f'{self.eval_name}.csv'
        self.csv = open(self.csv_path, 'w')
        self.csv.write('ImagePath,PredPath,Score,StatMasks\n')
        print(f'Saving results in {self.csv_path}')

    def write_result(self, d) :
        """Write a result line in the csv path

        Parameters
        ----------
        d : dict with the different values to write, containint.
            d['ImagePath'] (torch.tensor - str) : Image path for each image.
            d['PredPath'] (torch.tensor - str) : Pred path for each image.
            d['Score'] (torch.tensor - float) : Score for each image
        Returns
        -------
        type
            Description of returned object.

        """

        for i in range(len(d['ImagePath'])) :
            self.csv.write(f"{d['ImagePath'][i]},{d['PredPath'][i]},{d['Score'][i].item()},{json.dumps(d['StatMasks'][i], separators=('|',':'))}\n")

    def save_binary(self, d) :
        """Save Binary Mask with the prediction.

        Parameters
        ----------
        d : dict with the different values to write, containint.
            'ImagePath' (torch.tensor - str) : Image path for each image.
            'PredMask' (torch.tensor) : Binary segmentation mask ( b, i, j) with frgd : 1 and bkgd : 0
        """
        for i in range(len(d['ImagePath'])) :
            im = Image.fromarray(d['PredMask'][i].numpy())
            bin_path_save = self.path_save / 'BinaryMask' / d['ImagePath'][i]
            Path(bin_path_save).parent.mkdir(parents=True, exist_ok=True)
            im.save(Path(bin_path_save).with_suffix('.png'))

    def generate_fig(self,d) :
        """Generate and save figure with the probabilities and the choosen
        segments.

        Parameters
        ----------
        d : dict with the different values to write, containing.
            'ImagePath' (torch.tensor - str) : Image path for each image.
            'Score' (torch.tensor - float) : Score for each image
            'Flow' (torch.tensor - float) : Optical flow (b, 2, i, j)
            'Pred' (torch.tensor - float): Probability segmentation map (b, l, i, j) with l classes
            'GtMask' (torch.tensor - bool) : Ground Truth binary segmentation map (b, i, j)
            'PredMask' (torch.tensor) : Binary segmentation mask ( b, i, j) with frgd : 1 and bkgd : 0
        """
        for i in range(len(d['ImagePath'])) :
            K = d['Pred'][i].shape[0]
            fig, axs = plt.subplots(2, math.ceil(K / 2) + 2, figsize=(50,10))
            [a.axis('off') for a in axs.flatten()]
            axs[0,0].imshow(d['Pred'][i].argmax(0), cmap='Set1')
            axs[1,0].imshow(flowiz.convert_from_flow(d['Flow'][i].permute(1,2,0).numpy()))
            for j in range(K) :
                axs[j % 2, j // 2 +1].set_title(f'Layer : {j}')
                axs[j % 2, j // 2 +1].imshow(d['Pred'][i][j], vmin=0, vmax=1)
            axs[0, -1].set_title(f'PredMask : Score {d["Score"][i]:.2f}')
            axs[0, -1].imshow(d['PredMask'][i])
            axs[1, -1].set_title('GtMask')
            axs[1, -1].imshow(d['GtMask'][i])
            fig.tight_layout()
        img_path_save = self.path_save / d['ImagePath'][i]
        Path(img_path_save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(img_path_save).with_suffix('.jpg'))
        plt.close(fig)
