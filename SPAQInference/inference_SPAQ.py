import cv2
import torch
import torchvision
import onnxruntime as ort
from yacs.config import CfgNode as CN
import numpy as np
from PIL import Image
import os
import cv2
from .MT_A_demo import MTA
from .Prepare_image import Image_load

class Predictor():
    def __init__(self, weightsDir='.') -> None:
        model_path = os.path.join(weightsDir,'./MT-A_release.pt')
        self.model = MTA()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        checkpoint = torch.load(model_path,map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.prepare_image = Image_load(size=512, stride=224)



    def predict(self, img):
        patchs= self.prepare_image(img.convert("RGB"))
		#MOS	Brightness	Colorfulness	Contrast	Noisiness	Sharpness
        patchs = patchs.to(self.device)
        self.model.eval()
        MOS =self.model(patchs)[:, 0].mean().item()
        return {'SPAQ':MOS}

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
