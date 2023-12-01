import cv2
import torch
import torchvision
import onnxruntime as ort
from .utils.inference import inference
from .utils import get_config
from yacs.config import CfgNode as CN
import numpy as np
from PIL import Image
import os
import cv2
from easy_ViTPose import VitInference

class Predictor():
    def __init__(self, weightsDir='.') -> None:

        # set is_video=True to enable tracking in video inference
        # be sure to use VitInference.reset() function to reset the tracker after each video
        # There are a few flags that allows to customize VitInference, be sure to check the class definition
        model_path = os.path.join(weightsDir,'./vitpose-h-coco.pth')
        yolo_path = os.path.join(weightsDir,'./yolov8s.pt')

        # If you want to use MPS (on new macbooks) use the torch checkpoints for both ViTPose and Yolo
        # If device is None will try to use cuda -> mps -> cpu (otherwise specify 'cpu', 'mps' or 'cuda')
        # dataset and det_class parameters can be inferred from the ckpt name, but you can specify them.
        self.model = VitInference(model_path, yolo_path, model_name='h', yolo_size=320, is_video=False, device=None)



    def predict(self, img):
        # Infer keypoints, output is a dict where keys are person ids and values are keypoints (np.ndarray (25, 3): (y, x, score))
        # If is_video=True the IDs will be consistent among the ordered video frames.
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        keypoints = self.model.inference(img)

        # call model.reset() after each video

        img = self.model.draw(show_yolo=True)  # Returns RGB image with drawings        
        color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        # Displaying the converted image 
        img = Image.fromarray(color_coverted) 
        return img,keypoints


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
