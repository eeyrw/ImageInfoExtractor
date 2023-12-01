import torch
import torchvision
import easyocr
from PIL import Image
import os
import numpy as np

class Predictor():
    def __init__(self, weightsDir='.') -> None:
        self.reader = easyocr.Reader(['ch_sim','en'],model_storage_directory=os.path.join(weightsDir,'model')) # this needs to run only once to load the model into memory

    def predict(self, img):
        open_cv_image = np.array(img) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return self.reader.readtext(open_cv_image)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
