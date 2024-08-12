import easyocr
from PIL import Image
import os
import numpy as np



class Predictor():
    def __init__(self, weightsDir='.', device='cuda') -> None:
        self.reader = easyocr.Reader(['en','ja'],model_storage_directory=os.path.join(weightsDir,'model'),gpu=device) # this needs to run only once to load the model into memory

    def predict(self, img):
        width, height = img.size
        open_cv_image = np.array(img) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        results = self.reader.readtext(open_cv_image)
        resultsWithoutNumpy = []
        wh = np.asarray([width, height])
        for (p1,p2,p3,p4),text,score in results:
            if score>0.2:
                p1 = np.round(p1/wh,decimals=4).tolist()
                p2 = np.round(p2/wh,decimals=4).tolist()
                p3 = np.round(p3/wh,decimals=4).tolist()
                p4 = np.round(p4/wh,decimals=4).tolist()
                score = np.round(score,decimals=3).item()
                resultsWithoutNumpy.append(((p1,p2,p3,p4),text,score))
            
        return resultsWithoutNumpy


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
