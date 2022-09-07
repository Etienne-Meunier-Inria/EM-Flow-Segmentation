import torch, os
from DataLoadingModule import DataLoadingModule
import numpy as np
import cv2
import operator

class ScoreModule() :
    def __init__(self,scoring_method) :
        """
        Score Module involve giving a score to a binary prediction compared
        to potentially a GtMask
        """
        self.request = set()
        self.select_scoring_method(scoring_method)

    def select_scoring_method(self, scoring_method) :
        if scoring_method == 'db_eval_iou' :
            self.request.add('GtMask')
            self.scoring_method = self.db_eval_iou
        elif scoring_method == 'bbox_jacc' :
            self.request.add('GtMask')
            self.scoring_method = self.bbox_jacc
        else :
            raise Exception(f'Scoring method: {scoring_method} not implemented')


    def score(self, d) :
        """Assign a score to a prediction and return it.

        Parameters
        ----------
        d (dict) : dictionnary containing tensors for bianrisation, potentially :
            'Pred' (torch.tensor - float): Probability segmentation map (b, l, i, j) with l classes
            'GtMask' (torch.tensor - bool) : Ground Truth binary segmentation map (b, i, j)

        Returns
        -------
        None but add to dict 'Score' ( torch.tensor - float) : score for each element in the batch
        """
        assert d['PredMask'].dtype == torch.bool, f'Binary Mask should be bool currently is {d["PredMask"].dtype}'
        if 'GtMask' in self.request :
            assert d['GtMask'].dtype == torch.bool, f'Gt Mask should be bool currently is {d["GtMask"].dtype}'
        d['Score'] = self.scoring_method(**d)
        assert d['Score'].shape[0] == d['PredMask'].shape[0], f'Returned score is not the right shape : {d["Score"].shape}'

    def stat_masks(self, d) :
        """For each mask compute overlap and size of the mask.

        Parameters
        ----------
        d (dict) : dictionnary containing tensors for bianrisation, potentially :
            'Pred' (torch.tensor - float): Probability segmentation map (b, l, i, j) with l classes
            'GtMask' (torch.tensor - bool) : Ground Truth binary segmentation map (b, i, j)

        Returns
        -------
        None but add to dict 'StatMasks' ( dict) : including statistics for each mask and each element in batch
            'Overlap' (int) : Number of overlaping pixels between GtMask and Mask
            'Size' (int) : Size of the mask
        """
        stat_masks = []
        for bi in range(d['Pred'].shape[0]) :
            sm = {}
            GtMask = d['GtMask'][bi]
            Pred = d['Pred'][bi]
            argmap = Pred.argmax(0)
            sm['Size_Gt'] = GtMask.sum().item()
            for l in range(Pred.shape[0]) :
                sm[f'Overlap_{l}'] = ((argmap == l)  & GtMask).sum().item()
                sm[f'Size_{l}'] = (argmap == l).sum().item()
            stat_masks.append(sm)
        d['StatMasks'] = stat_masks

    @staticmethod
    def db_eval_iou(GtMask,PredMask,**k):
        """
        Compute region similarity as the Jaccard Index.

        Arguments:
        GtMask   (torch - bool): binary annotation map. (b, I, J)
        PredMask (torch - bool): binary segmentation map. (b, I, J)

        Return:
        jaccard (float): region similarity (b)
        """
        assert GtMask.dtype == torch.bool, f'GtMask is not bool : {GtMask.dtype}'
        assert PredMask.dtype == torch.bool, f'PredMask is not bool : {PredMask.dtype}'

        k = (PredMask & GtMask).sum(axis=(1,2)) / (PredMask | GtMask).sum(axis=(1,2))
        k[k.isnan()] = 1 # 1 jaccard score if both gt and pred are empty
        return k

    @staticmethod
    def bbox_jacc(GtMask, PredMask, **k) :
        """
        First draw the smallest box around the foreground then compute the Jaccard

        Arguments :
        GtMask   (torch - bool): binary annotation map. (b, I, J)
        PredMask (torch - bool): binary segmentation map. (b, I, J)

        Return:
        jaccard (float): region similarity (b)
        """
        b = PredMask.shape[0]
        boxmasks = []
        for i in range(b) :
            boxmasks.append(ScoreModule.get_bbox_mask(PredMask[0].numpy()))
        PredMask[:,:,:] = torch.tensor(np.stack(boxmasks)).to(bool) # Inplace update stored PredMask
        return ScoreModule.db_eval_iou(GtMask, PredMask)

    @staticmethod
    def get_bbox_mask(bmask) :
        """
        Get a binary mask ( one sample ) and extract a bounding box max with the bounding
        box around the biggest countour. Based on the method from motion grouping
        (Zisserman, 2021) paper.
        Arguments :
        bmask (ndarray) : binary mask with the foregroun at 1 : {0,1}

        Returns :
        boxmask (ndarray) : binary mask with the foreground box at 1 : {0,1}
        """
        bmask = (bmask * 255).astype(np.uint8)
        boxmask = np.zeros_like(bmask, dtype=bool)
        if bmask.max() > 0 :
            contours = cv2.findContours(bmask.astype(
                            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            area = 0

            for cnt in contours:
                (x_, y_, w_, h_) = cv2.boundingRect(cnt)
                if w_*h_ > area:
                    x = x_
                    y = y_
                    w = w_
                    h = h_
                    area = w_ * h_
            boxmask[y:y+h, x:x+w] = True
        return boxmask
