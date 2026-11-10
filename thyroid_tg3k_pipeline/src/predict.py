import numpy as np
import torch
from src.data import letterbox,tensor,restore

@torch.inference_mode()
def predict(model,images,side,device):
    prepared=[letterbox(im,side) for im in images]
    x=torch.stack([tensor(p[0]) for p in prepared]).to(device)
    probability=model(x).sigmoid()[:,0].cpu().numpy()
    return [restore(p,data[3]) for p,data in zip(probability,prepared)]

def reconcile(original,probability,label,threshold=.5,min_area=1):
    if original.shape!=probability.shape:
        raise ValueError('Prediction not in original image coordinates')
    mask=probability>=threshold
    if np.count_nonzero(mask)<min_area:
        mask=np.zeros_like(mask)
    if np.any(original):
        return original.copy(),'provided',mask
    if mask.any():
        return mask.astype(np.uint8)*label,'pseudo',mask
    return original.copy(),'unknown',mask
