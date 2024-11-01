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


class Predictor():
    def __init__(self, weightsDir='.') -> None:
        self.cfg = CN._load_cfg_py_source('ViTPoseInference/configs/custom_config.py')
        self.cfg.cpu = True
        if self.cfg.cpu:
            EP_list = ['CPUExecutionProvider']
        else:
            EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        YOLOV6_PATH = os.path.join(weightsDir,'yolov6m.onnx')
        VITPOSE_PATH = os.path.join(weightsDir,'vitpose-b-multi-coco.onnx')
        self.yolov6_sess = ort.InferenceSession(YOLOV6_PATH, providers=EP_list)
        self.vitpose_sess = ort.InferenceSession(VITPOSE_PATH, providers=EP_list)

    def predict(self, img):
        img_origin = img
        img_origin = np.expand_dims(img_origin, axis=0)
        imgs, pred = inference(img_origin, self.yolov6_sess, self.vitpose_sess, self.cfg)
        color_coverted = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB) 
        # Displaying the converted image 
        img = Image.fromarray(color_coverted) 
        # print('-'*10 + "\nPress 'Q' key on OpenCV window if you want to close")
        # cv2.imshow("OpenCV", img[0])

        # if cfg.save:
        #     save_name = img_path.replace(".jpg", "_result.jpg")
        #     cv2.imwrite(save_name, img[0])
        # if cfg.save_prediction:
        #     preds = {'bbox':[], 'pose':[]}
        #     preds['bbox'].extend(pred[0])
        #     preds['pose'].extend(pred[1])
        #     save_name = img_path.replace(".jpg", "_prediction.pkl")
        #     with open(save_name, 'wb') as f:
        #         pickle.dump(preds, f)
        preds = {'bbox':[], 'pose':[]}
        preds['bbox'].extend(pred[0])
        preds['pose'].extend(pred[1])
        return img,preds


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

