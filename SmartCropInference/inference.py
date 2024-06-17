from __future__ import annotations
import os

from .smartcrop import SmartCropWithFace

class Predictor():
    def __init__(self, weightsDir='.', device='cuda:6') -> None:
        self.device = device

        self.cropper = SmartCropWithFace(debug=False,
                                          faceDetWeight = os.path.join(weightsDir,'face_detection_yunet_2023mar.onnx'))


    def predict(self, raw_image):

        ret = self.cropper.crop(raw_image, 100, 100,max_scale=1.0,min_scale=1.0,step=4,prescale=True)
        raw_w,raw_h = raw_image.size
        return {'A_CENTER': (float('%.4f'%((ret['top_crop']['x']+ret['top_crop']['width']/2)/raw_w)),
                              float('%.4f'%((ret['top_crop']['y']+ret['top_crop']['height']/2)/raw_h))
                              )}
    

