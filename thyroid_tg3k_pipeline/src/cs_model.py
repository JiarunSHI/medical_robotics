"""CS CMU-Net inference using the exact original RGB/[0,1] letterbox transform."""
import numpy as np
from PIL import Image,ImageOps
import torch
from torch.torch_version import TorchVersion
from vendor.cmunet_cs.CMUNet import CMUNet
from src.data import restore

def load_cs(path,device):
    # Original metadata contains torch.__version__ as TorchVersion, not just str.
    with torch.serialization.safe_globals([TorchVersion]):
        ckpt=torch.load(path,map_location='cpu',weights_only=True)
    cfg=ckpt['config']
    if cfg.get('target')!='cs' or cfg.get('architecture')!='Official CMUNet l=7 k=7' or cfg.get('smoke'):
        raise ValueError('Expected trained CS CMUNet l=7 k=7 checkpoint')
    model=CMUNet(img_ch=3,output_ch=1,l=7,k=7).to(device)
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    return model,ckpt

def prepare_cs(image):
    image=image.convert('RGB');w,h=image.size
    size=(round(w*256/max(w,h)),round(h*256/max(w,h)))
    left,top=(256-size[0])//2,(256-size[1])//2
    image=image.resize(size,Image.Resampling.BILINEAR)
    image=ImageOps.expand(image,(left,top,256-size[0]-left,256-size[1]-top))
    x=torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1).copy())/255
    return x,(left,top,*size,w,h)

@torch.inference_mode()
def predict_cs(model,images,device):
    prepared=[prepare_cs(im) for im in images]
    x=torch.stack([r[0] for r in prepared]).to(device)
    probs=model(x).float().sigmoid()[:,0].cpu().numpy()
    return [restore(prob,data[1]) for prob,data in zip(probs,prepared)]
