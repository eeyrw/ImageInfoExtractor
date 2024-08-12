import cv2
from rtmlib import Body, Wholebody, draw_skeleton
from PIL import Image
import os
import numpy as np


class Predictor():

    def __init__(self, weightsDir='.', device='cpu') -> None:
        backend = 'onnxruntime'  # opencv, onnxruntime, openvino
        openpose_skeleton = False  # True for openpose-style, False for mmpose-style
        # self.model = Wholebody(
        #     to_openpose=openpose_skeleton,
        #     # pose=os.path.join(weightsDir, 'dw-ll_ucoco_384.onnx'),
        #     mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
        #     backend=backend,
        #     device=device)
        self.model = Body(
            to_openpose=openpose_skeleton,
            # pose=os.path.join(weightsDir, 'dw-ll_ucoco_384.onnx'),
            mode='performance',  # 'performance', 'lightweight', 'balanced'. Default: 'balanced'
            backend=backend,
            device=device)

    def predict(self, img, debug=False):
        width, height = img.size
        open_cv_image = np.array(img)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        keypoints, scores, bboxes = self.model(open_cv_image,
                                               return_bboxes=True)
        if len(bboxes)!=0:
            kpt_thr = 0.3
            if debug:
                img_show = open_cv_image.copy()
                img_show = draw_skeleton(img_show,
                                        keypoints,
                                        scores,
                                        openpose_skeleton=False,
                                        kpt_thr=kpt_thr,
                                        line_width=2)
                cv2.imwrite('c.jpg', img_show)

            keypoints_mapped = []
            for instance_kpts, instance_scores, instance_bbox in zip(
                    keypoints, scores, bboxes):
                invalidKeypointsIdx = np.where(instance_scores < kpt_thr)[0]
                instance_kpts[invalidKeypointsIdx] = -1
                instance_kpts_mapped_x = [
                    round(float(kpt[0]) / width, ndigits=3)
                    for kpt in instance_kpts
                ]
                instance_kpts_mapped_y = [
                    round(float(kpt[1]) / height, ndigits=3)
                    for kpt in instance_kpts
                ]
                keypoints_mapped.append({
                    'BBOX': (
                        round(float(instance_bbox[0]) / width, ndigits=3),  # x1
                        round(float(instance_bbox[1]) / height, ndigits=3),  # y1
                        round(float(instance_bbox[2]) / width, ndigits=3),  # x2
                        round(float(instance_bbox[3]) / height, ndigits=3),  # y2
                    ),
                    'INVLD_KPTS_IDX': [int(idx) for idx in invalidKeypointsIdx],
                    'KPTS_X':
                    instance_kpts_mapped_x,
                    'KPTS_Y':
                    instance_kpts_mapped_y,
                })
        else:
            keypoints_mapped =[]
        return {'POSE_KPTS': keypoints_mapped}


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


if __name__ == "__main__":
    pr = Predictor(weightsDir='DLToolWeights')
    with open('aa.jpg', 'rb') as f:
        imgs = Image.open(f).convert('RGB')
        print(pr.predict(imgs, debug=True
                         ))
