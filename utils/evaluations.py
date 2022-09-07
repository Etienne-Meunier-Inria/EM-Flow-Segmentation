import torch
import numpy as np
import cv2

def db_eval_iou(annotation,segmentation):
    """
    Compute region similarity as the Jaccard Index.

    Arguments:
    annotation   (torch): binary annotation   map. (b, I, J)
    segmentation (ndarray): binary segmentation map. (b, I, J)

    Return:
    jaccard (float): region similarity (b)
    """
    annotation = annotation.to(torch.bool)
    segmentation = segmentation.to(torch.bool)

    k = (segmentation & annotation).sum(axis=(1,2)) / (segmentation | annotation).sum(axis=(1,2))
    k[k.isnan()] = 1 # 1 jaccard score if both gt and pred are empty
    return k
