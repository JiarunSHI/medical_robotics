import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance
from torch.utils.data import Dataset

def letterbox(image, side, mask=None):
    image = image.convert('L').convert('RGB')
    w,h = image.size
    scale = side / max(w,h)
    nw,nh = max(1,round(w*scale)),max(1,round(h*scale))
    left,top = (side-nw)//2,(side-nh)//2
    padding = (left,top,side-nw-left,side-nh-top)
    image = ImageOps.expand(image.resize((nw,nh),Image.Resampling.BILINEAR),padding)
    valid = ImageOps.expand(Image.new('L',(nw,nh),255),padding)
    if mask is not None:
        if mask.size != (w,h):
            raise ValueError('Image/mask size mismatch')
        mask = ImageOps.expand(mask.resize((nw,nh),Image.Resampling.NEAREST),padding)
    return image,mask,valid,(left,top,nw,nh,w,h)

def tensor(image):
    x = np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()/255
    return (torch.from_numpy(x)-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def restore(probability, geometry):
    left,top,nw,nh,w,h = geometry
    cropped = probability[top:top+nh,left:left+nw]
    return np.asarray(Image.fromarray(cropped.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))

class Frames(Dataset):
    def __init__(self, rows, side, augment=False, pseudo_weight=.5):
        self.rows,self.side,self.augment,self.pseudo_weight=rows,side,augment,pseudo_weight
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(r['image']) as im, Image.open(r['mask']) as ma:
            # JPG TG3K masks must be thresholded, not >0 (compression artifacts).
            mask=Image.fromarray((np.asarray(ma.convert('L'))>=128).astype(np.uint8)*255)
            image,mask,valid,_=letterbox(im,self.side,mask)
        if self.augment:
            if random.random()<.5:
                image,mask,valid=map(ImageOps.mirror,(image,mask,valid))
            angle=random.uniform(-8,8)
            image=image.rotate(angle,Image.Resampling.BILINEAR)
            mask=mask.rotate(angle,Image.Resampling.NEAREST)
            valid=valid.rotate(angle,Image.Resampling.NEAREST)
            image=ImageEnhance.Brightness(image).enhance(random.uniform(.9,1.1))
        y=torch.from_numpy((np.asarray(mask)>0).astype(np.float32)[None])
        v=torch.from_numpy((np.asarray(valid)>0).astype(np.float32)[None])
        weight=self.pseudo_weight if r.get('source')=='pseudo' else 1.
        return tensor(image),y,v,str(r['case_id']),weight
