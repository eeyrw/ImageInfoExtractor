import cv2
from rtmlib import Wholebody, draw_skeleton
from PIL import Image
import os
import numpy as np

class Predictor():
    def __init__(self, weightsDir='.') -> None:

        device = 'cuda:6'  # cpu, cuda, mps
        backend = 'onnxruntime'  # opencv, onnxruntime, openvino

        openpose_skeleton = False  # True for openpose-style, False for mmpose-style

        self.model = Wholebody(to_openpose=openpose_skeleton,
                            mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
                            backend=backend, device=device)

    def predict(self, img):
        width, height = img.size
        open_cv_image = np.array(img) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        keypoints, scores, bboxes = self.model(open_cv_image,return_bboxes=True)
        kpt_thr = 0.5
        keypoints_mapped=[]
        for instance_kpts,instance_scores,instance_bbox in zip(keypoints,scores,bboxes):
            instance_kpts_mapped = [
                (round(float(kpt[0])/width, ndigits=3),round(float(kpt[1])/height, ndigits=3)) if score>=kpt_thr else (-1,-1)
                for kpt,score in zip(instance_kpts,instance_scores)
            ]
            keypoints_mapped.append({'BBOX':(
                round(float(instance_bbox[0])/width, ndigits=3), #x1
                round(float(instance_bbox[1])/height, ndigits=3), #y1
                round(float(instance_bbox[2])/width, ndigits=3), #x2
                round(float(instance_bbox[3])/height, ndigits=3), #y2
            ),'KPTS':instance_kpts_mapped})
        return {'POSE_KPTS':keypoints_mapped}


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

if __name__ == "__main__":
    pr = Predictor(weightsDir='DLToolWeights')
    with open('aa.jpg', 'rb') as f:
        imgs = Image.open(f).convert('RGB')
        print(pr.predict(imgs))
