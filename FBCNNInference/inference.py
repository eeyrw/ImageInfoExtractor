import torch
import torchvision
from PIL import Image
import os
from .models.network_fbcnn import FBCNN as net
import numpy as np

class Predictor():
    def __init__(self, weightsDir='.') -> None:
        n_channels = 3            # set 1 for grayscale image, set 3 for color image
        model_path = os.path.join(weightsDir, 'fbcnn_color.pth')
        nc = [64, 128, 256, 512]
        nb = 4
        self.model = net(in_nc=n_channels, out_nc=n_channels,
                         nc=nc, nb=nb, act_mode='R')
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.to('cuda').half()
        self.model.eval()

    def predict(self, img):

        img_L = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)
        img_L = img_L.to('cuda',dtype=torch.float16)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        #img_E,QF = model(img_L, torch.tensor([[0.6]]))
        QF = self.model(img_L)
        QF = 1 - QF
        QF = round(float(QF*100))
        # logger.info('predicted quality factor: {:d}'.format(round(float(QF*100))))
        # util.imsave(img_E, os.path.join(E_path, img_name+'.png'))
        return {'QF': QF}


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
