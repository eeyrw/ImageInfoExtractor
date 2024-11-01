import itertools
import cv2
from ultralytics import YOLO
from PIL import Image
import os
import numpy as np
import torch
from torchvision import transforms as T
class Predictor():

    def __init__(self, weightsDir='.',weightName=None, device='cpu') -> None:
        model = YOLO(os.path.join(weightsDir,'yolo',weightName))  # pretrained YOLO11n model
        self.classNameDict = model.names
        self.device = device
        self.model = model.to(self.device)
        self.transform = None

    def predict(self, img, debug=False):
        results = self.model(img,verbose=False)
        finalOut = []
        for result in results:
            clsList = result.boxes.cls.reshape(-1,1)
            confList = result.boxes.conf.reshape(-1,1)
            xywhnList = result.boxes.xywhn
            ccxywhnList = torch.concat((confList,clsList,xywhnList),1).cpu().numpy().round(5)
            finalOut.append(ccxywhnList)
        return {'OBJS':{'CLS':self.classNameDict,'RESULTS':finalOut}}
    
    def predict_batch(self, imgs):
        with torch.no_grad():
            resultss = self.model(imgs,verbose=False)
            finalOuts=[]
            for results in resultss:
                finalOut = []
                for result in results:
                    clsList = result.boxes.cls.reshape(-1,1)
                    confList = result.boxes.conf.reshape(-1,1)
                    xywhnList = result.boxes.xywhn
                    ccxywhnList = torch.concat((confList,clsList,xywhnList),1).cpu().numpy().round(5)
                    finalOut.append(ccxywhnList)    
                finalOuts.append({'OBJS':{'CLS':self.classNameDict,'RESULTS':finalOut}})
        return finalOuts


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == "__main__":
    from pillow_heif import register_heif_opener
    register_heif_opener()
    pr = Predictor(weightsDir='DLToolWeights',weightName='pword.pt')
    with open('dualP.webp', 'rb') as f:
        imgs = Image.open(f).convert('RGB')
        print(pr.predict(imgs, debug=True
                         ))
