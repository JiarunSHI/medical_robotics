import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

ARCHITECTURE = 'mobilenet_v2_unet_v1'

class Block(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

class MobileUNet(nn.Module):
    """MobileNetV2 encoder, five-scale skip connections, U-Net decoder."""
    def __init__(self, imagenet=False):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if imagenet else None
        self.encoder = mobilenet_v2(weights=weights).features
        self.blocks = nn.ModuleList([Block(1280+96,256), Block(256+32,128),
                                     Block(128+24,64), Block(64+16,32)])
        self.final = nn.Sequential(Block(32,16), nn.Conv2d(16,1,1))

    def forward(self, x):
        size = x.shape[-2:]
        skips = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (1,3,6,13,18):
                skips.append(x)
        x = skips[-1]
        for block, skip in zip(self.blocks, reversed(skips[:-1])):
            x = block(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip],1))
        return self.final(F.interpolate(x,size=size,mode='bilinear',align_corners=False))

def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != ARCHITECTURE:
        raise ValueError('Checkpoint architecture differs')
    if checkpoint.get('preprocessing') != 'gray_rgb_imagenet_letterbox_v1':
        raise ValueError('Unsupported checkpoint preprocessing')
    model = MobileUNet().to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, checkpoint
