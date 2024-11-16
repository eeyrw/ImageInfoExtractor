import json
import math
from time import time
import PIL
import cv2
from matplotlib import pyplot as plt
import numpy as np
from rtmlib import draw_mmpose, draw_bbox
import os
import random
from PIL import Image, ImageFont, ImageDraw
from scipy.spatial import distance_matrix
from pillow_heif import register_heif_opener
from scipy.optimize import linear_sum_assignment
from MultiDatasetExtractor import MultiDatasetExtractor

register_heif_opener()

male20 = dict(name='male20',
              keypoint_info={
                  0:
                  dict(name='nose', id=0, color=[51, 153, 255], swap=''),
                  1:
                  dict(name='left_eye',
                       id=1,
                       color=[51, 153, 255],
                       swap='right_eye'),
                  2:
                  dict(name='right_eye',
                       id=2,
                       color=[51, 153, 255],
                       swap='left_eye'),
                  3:
                  dict(name='left_ear',
                       id=3,
                       color=[51, 153, 255],
                       swap='right_ear'),
                  4:
                  dict(name='right_ear',
                       id=4,
                       color=[51, 153, 255],
                       swap='left_ear'),
                  5:
                  dict(name='left_shoulder',
                       id=5,
                       color=[0, 255, 0],
                       swap='right_shoulder'),
                  6:
                  dict(name='right_shoulder',
                       id=6,
                       color=[255, 128, 0],
                       swap='left_shoulder'),
                  7:
                  dict(name='left_elbow',
                       id=7,
                       color=[0, 255, 0],
                       swap='right_elbow'),
                  8:
                  dict(name='right_elbow',
                       id=8,
                       color=[255, 128, 0],
                       swap='left_elbow'),
                  9:
                  dict(name='left_wrist',
                       id=9,
                       color=[0, 255, 0],
                       swap='right_wrist'),
                  10:
                  dict(name='right_wrist',
                       id=10,
                       color=[255, 128, 0],
                       swap='left_wrist'),
                  11:
                  dict(name='left_hip',
                       id=11,
                       color=[0, 255, 0],
                       swap='right_hip'),
                  12:
                  dict(name='right_hip',
                       id=12,
                       color=[255, 128, 0],
                       swap='left_hip'),
                  13:
                  dict(name='left_knee',
                       id=13,
                       color=[0, 255, 0],
                       swap='right_knee'),
                  14:
                  dict(name='right_knee',
                       id=14,
                       color=[255, 128, 0],
                       swap='left_knee'),
                  15:
                  dict(name='left_ankle',
                       id=15,
                       color=[0, 255, 0],
                       swap='right_ankle'),
                  16:
                  dict(name='right_ankle',
                       id=16,
                       color=[255, 128, 0],
                       swap='left_ankle'),
                  17:
                  dict(name='dick_root', id=17, color=[8, 218, 183], swap=''),
                  18:
                  dict(name='dick_mid', id=18, color=[8, 218, 14], swap=''),
                  19:
                  dict(name='dick_head',
                       id=19,
                       color=[179, 218, 8],
                       swap=''),
                  20:
                  dict(name='ball_left',
                       id=20,
                       color=[232, 110, 23],
                       swap='ball_right'),
                  21:
                  dict(name='ball_right',
                       id=21,
                       color=[130, 37, 226],
                       swap='ball_left')
              },
              skeleton_info={
                  0:
                  dict(link=('left_ankle', 'left_knee'),
                       id=0,
                       color=[0, 255, 0]),
                  1:
                  dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255,
                                                                    0]),
                  2:
                  dict(link=('right_ankle', 'right_knee'),
                       id=2,
                       color=[255, 128, 0]),
                  3:
                  dict(link=('right_knee', 'right_hip'),
                       id=3,
                       color=[255, 128, 0]),
                  4:
                  dict(link=('left_hip', 'right_hip'),
                       id=4,
                       color=[51, 153, 255]),
                  5:
                  dict(link=('left_shoulder', 'left_hip'),
                       id=5,
                       color=[51, 153, 255]),
                  6:
                  dict(link=('right_shoulder', 'right_hip'),
                       id=6,
                       color=[51, 153, 255]),
                  7:
                  dict(link=('left_shoulder', 'right_shoulder'),
                       id=7,
                       color=[51, 153, 255]),
                  8:
                  dict(link=('left_shoulder', 'left_elbow'),
                       id=8,
                       color=[0, 255, 0]),
                  9:
                  dict(link=('right_shoulder', 'right_elbow'),
                       id=9,
                       color=[255, 128, 0]),
                  10:
                  dict(link=('left_elbow', 'left_wrist'),
                       id=10,
                       color=[0, 255, 0]),
                  11:
                  dict(link=('right_elbow', 'right_wrist'),
                       id=11,
                       color=[255, 128, 0]),
                  12:
                  dict(link=('left_eye', 'right_eye'),
                       id=12,
                       color=[51, 153, 255]),
                  13:
                  dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
                  14:
                  dict(link=('nose', 'right_eye'), id=14, color=[51, 153,
                                                                 255]),
                  15:
                  dict(link=('left_eye', 'left_ear'),
                       id=15,
                       color=[51, 153, 255]),
                  16:
                  dict(link=('right_eye', 'right_ear'),
                       id=16,
                       color=[51, 153, 255]),
                  17:
                  dict(link=('left_ear', 'left_shoulder'),
                       id=17,
                       color=[51, 153, 255]),
                  18:
                  dict(link=('right_ear', 'right_shoulder'),
                       id=18,
                       color=[51, 153, 255]),
                  19:
                  dict(link=('dick_root', 'dick_mid'),
                       id=19,
                       color=[206, 18, 121]),
                  20:
                  dict(link=('dick_mid', 'dick_head'),
                       id=20,
                       color=[206, 18, 121]),
                  21:
                  dict(link=('ball_left', 'dick_root'),
                       id=21,
                       color=[206, 18, 121]),
                  22:
                  dict(link=('ball_right', 'dick_root'),
                       id=22,
                       color=[206, 18, 121])
              })


class PoseDsCreator:

    def __init__(self, dsPath) -> None:
        self.dsPath = dsPath

    def constantAreaResize(self, w, h, area):
        resize_ratio = math.sqrt(area / (w * h))
        return tuple(math.ceil(x * resize_ratio) for x in (w, h))

    def genImageJiasaw(self, imageList, width, height, col, row, outPath):
        to_image = Image.new('RGB', (col * width, row * height))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(row):
            for x in range(col):
                from_image = imageList[y * col + x]
                to_image.paste(from_image, (x * width, y * height))
        return to_image.save(outPath, quality=90)  # 保存新图

    def __len__(self) -> int:
        return len(self.imagesIdxList)

    def resample_ds_by_weight(self):
        print('Resample DS')
        rawImagesIdxList = list(range(len(self.imageInfoList)))
        rng = np.random.default_rng()
        resampleList = rng.choice(rawImagesIdxList,
                                  8888,
                                  replace=False,
                                  p=self.imagesWeightList)
        self.imagesIdxList = resampleList.tolist()

    def generate_color_dict(self, class_dict):
        # 预定义20种颜色
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                  (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                  (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
                  (192, 192, 192), (128, 128, 128), (64, 0, 0), (0, 64, 0),
                  (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64)]

        color_dict = {}
        for i, class_name in enumerate(class_dict.keys()):
            color_dict[class_name] = colors[i % len(colors)]  # 循环使用颜色

        return color_dict

    def generateKeyPoint(self, anno):
        obs = anno['OBJS']
        classToId = {v: int(k) for k, v in obs['CLS'].items()}
        dickBboxes = self.xywhnTox1y1x2y2n([
            ccxywhn[2:6] for ccxywhn in filter(
                lambda bbox: bbox[1] == classToId['dick'] and bbox[0] > 0.3,
                obs['RESULTS'])
        ])
        ballBboxes = self.xywhnTox1y1x2y2n([
            ccxywhn[2:6] for ccxywhn in filter(
                lambda bbox: bbox[1] == classToId['balls'] and bbox[0] > 0.3,
                obs['RESULTS'])
        ])
        dickHeadBboxes = self.xywhnTox1y1x2y2n([
            ccxywhn[2:6] for ccxywhn in filter(
                lambda bbox: bbox[1] == classToId['dick-head'] and bbox[0] >
                0.3, obs['RESULTS'])
        ])

        dickAnchorList = []
        dickAnchorNum = 8
        dickSetDict = dict()
        for dickIdx, (x1, y1, x2, y2) in enumerate(dickBboxes):
            dickAnchorList.extend((x1, y1, (x1+x2)/2, y1,
                                   x2, y1, x2, (y1+y2)/2,
                                   x2, y2, (x1+x2)/2, y2,
                                   x1, y2, x1, (y1+y2)/2))

        dickAnchorList = np.asarray(dickAnchorList).reshape((-1, 2))
        ballCenterList = np.asarray(
            ballBboxes).reshape((-1, 2, 2)).mean(axis=1)
        dickHeadCenterList = np.asarray(
            dickHeadBboxes).reshape((-1, 2, 2)).mean(axis=1)

        kptLHip = 11
        kptRHip = 12
        bodyDickAttachPointList = []
        if len(anno['POSE_KPTS']) > 0:
            for poseDict in anno['POSE_KPTS']:
                x1, y1, x2, y2 = poseDict['BBOX']
                if len(poseDict['KPTS_X']) > 17:
                    poseDict['KPTS_X'] = poseDict['KPTS_X'][0:17]
                    poseDict['KPTS_Y'] = poseDict['KPTS_Y'][0:17]

                newInvalidKptsIdxList = []
                for newIdx in poseDict['INVLD_KPTS_IDX']:
                    if newIdx < 17:
                        newInvalidKptsIdxList.append(newIdx)
                poseDict['INVLD_KPTS_IDX'] = newInvalidKptsIdxList

                bodyAttachPoint = ((x1 + x2) / 2, (y1 + y2) / 2)
                if kptLHip not in poseDict[
                        'INVLD_KPTS_IDX'] and kptRHip not in poseDict[
                            'INVLD_KPTS_IDX']:
                    bodyAttachPoint = ((poseDict['KPTS_X'][kptLHip] +
                                        poseDict['KPTS_X'][kptRHip]) / 2,
                                       (poseDict['KPTS_Y'][kptLHip] +
                                        poseDict['KPTS_Y'][kptRHip]) / 2)
                bodyDickAttachPointList.extend(bodyAttachPoint)

        bodyDickAttachPointList = np.asarray(
            bodyDickAttachPointList).reshape((-1, 2))

        if len(dickBboxes) > 0:
            distanceMatP2D = distance_matrix(
                dickAnchorList, bodyDickAttachPointList)
            dickAnchorIdcs, personDickPosIdcs = linear_sum_assignment(
                distanceMatP2D)
            for dickAnchorIdx, personDickPosIdx in zip(dickAnchorIdcs, personDickPosIdcs):
                dickIdx = dickAnchorIdx//dickAnchorNum
                dickSetDict.setdefault(dickIdx, {'bodyIdx': personDickPosIdx,
                                                 'root': dickAnchorList[dickAnchorIdx],
                                                 'head': None, 'balls': []})

            if len(ballBboxes) > 0:
                distanceMatD2B = distance_matrix(
                    dickAnchorList, ballCenterList)
                dickAnchorIdcs, ballIdcs = linear_sum_assignment(
                    distanceMatD2B)
                dickBallAttachPointListDict = dict()
                for dickAnchorIdx, ballIdx in zip(dickAnchorIdcs, ballIdcs):
                    dickIdx = dickAnchorIdx//dickAnchorNum
                    if dickIdx in dickSetDict.keys():
                        dickSetDict[dickIdx]['balls'].append(
                            ballCenterList[ballIdx])
                        dickBallAttachPointListDict.setdefault(
                            dickIdx, []).append(dickAnchorList[dickAnchorIdx])

                for dickIdx, balls in dickBallAttachPointListDict.items():
                    dickSetDict[dickIdx]['root'] = np.asarray(
                        balls).mean(axis=0)

            if len(dickHeadBboxes) > 0:
                distanceMatD2H = distance_matrix(
                    dickAnchorList, dickHeadCenterList)
                dickAnchorIdcs, dickHeadIdcs = linear_sum_assignment(
                    distanceMatD2H)
                for dickAnchorIdx, dickHeadIdx in zip(dickAnchorIdcs, dickHeadIdcs):
                    dickIdx = dickAnchorIdx//dickAnchorNum
                    if dickIdx in dickSetDict.keys():  # only assign to alid dick
                        dickSetDict[dickIdx]['head'] = dickHeadCenterList[dickHeadIdx]

            for dickIdx, dickInfo in dickSetDict.items():
                if dickInfo['head'] is None:
                    singleDickAnchorList = dickAnchorList[dickIdx*dickAnchorNum:(
                        dickIdx+1)*dickAnchorNum]+1e-5
                    distanceMatR2D = np.reciprocal(distance_matrix(
                        singleDickAnchorList, np.asarray([dickInfo['root']])))
                    singleDickAnchorIdcs, rootIdcs = linear_sum_assignment(
                        distanceMatR2D)
                    dickSetDict[dickIdx]['head'] = singleDickAnchorList[singleDickAnchorIdcs[0]]

            for dickIdx, dickInfo in dickSetDict.items():
                invalidIdxList = []
                if dickInfo['bodyIdx'] is not None:
                    ball_left = dickInfo['root']+(-0.1, 0.1)
                    ball_right = dickInfo['root']+(0.1, 0.1)
                    if len(dickInfo['balls']) < 1:
                        invalidIdxList.extend((20, 21))
                    elif len(dickInfo['balls']) == 1:
                        ball_left = dickInfo['balls'][0]
                        invalidIdxList.extend((21,))
                    elif len(dickInfo['balls']) >= 2:
                        ball_left = dickInfo['balls'][0]
                        ball_right = dickInfo['balls'][1]

                    dick_root = dickInfo['root']
                    dick_mid = (dickInfo['head']+dickInfo['root'])/2
                    dick_head = dickInfo['head']

                    anno['POSE_KPTS'][dickInfo['bodyIdx']
                                      ]['INVLD_KPTS_IDX'].extend(invalidIdxList)
                    anno['POSE_KPTS'][dickInfo['bodyIdx']]['KPTS_X'].extend(
                        [dick_root[0], dick_mid[0], dick_head[0], ball_left[0], ball_right[0]])
                    anno['POSE_KPTS'][dickInfo['bodyIdx']]['KPTS_Y'].extend(
                        [dick_root[1], dick_mid[1], dick_head[1], ball_left[1], ball_right[1]])

        for bodyIdx, poseDict in enumerate(anno['POSE_KPTS']):
            if len(poseDict['KPTS_X']) == 17:
                possbileDickRoot = bodyDickAttachPointList[bodyIdx]
                possbileDickHead = possbileDickRoot+(0, 0.1)
                possbileDickMid = (possbileDickRoot+possbileDickHead)/2
                possibleBallLeft = possbileDickRoot+(-0.01, 0.01)
                possibleBallRight = possbileDickRoot+(0.01, 0.01)
                poseDict['KPTS_X'].extend((possbileDickRoot[0], possbileDickMid[0],
                                          possbileDickHead[0], possibleBallLeft[0], possibleBallRight[0]))
                poseDict['KPTS_Y'].extend((possbileDickRoot[1], possbileDickMid[1],
                                          possbileDickHead[1], possibleBallLeft[1], possibleBallRight[1]))
                poseDict['INVLD_KPTS_IDX'].extend((17, 18, 19, 20, 21))
        print(anno['IMG'], dickSetDict)
        return anno['POSE_KPTS']

    def xywhnTox1y1x2y2n(self, xywhnList):
        x1y1x2y2Listn = []
        for xywh in xywhnList:
            x, y, w, h = xywh
            x1 = (x - w / 2)
            y1 = (y - h / 2)
            x2 = (x + w / 2)
            y2 = (y + h / 2)
            x1y1x2y2Listn.append((x1, y1, x2, y2))
        return x1y1x2y2Listn

    def load_font(self, base_font_size):
        # 尝试加载 Arial 字体，如果不存在则使用默认字体
        try:
            font_path = "arial.ttf"
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, base_font_size)
        except IOError:
            pass
        return ImageFont.load_default()

    def drawSkleton(self, image, anno):
        width, height = image.size
        base_line_width = max(1, max(width, height) // 200)  # 最小线宽为1
        opencv_image = np.array(image)
        # Convert RGB to BGR
        opencv_image = opencv_image[:, :, ::-1].copy()
        bboxes = [(item['BBOX'][0] * width, item['BBOX'][1] * width,
                   item['BBOX'][2] * height, item['BBOX'][3] * height)
                  for item in anno]
        opencv_image = draw_bbox(opencv_image, bboxes)
        multipleKpts = []
        multipleScores = []
        for item in anno:
            kpts = []
            kptsScore = []
            for i, (x, y) in enumerate(
                    zip(item['KPTS_X'], item['KPTS_Y'])):
                kpts.append((x * width, y * height))
                if i in item['INVLD_KPTS_IDX']:
                    kptsScore.append(0)
                else:
                    kptsScore.append(1)
            multipleKpts.append(kpts)
            multipleScores.append(kptsScore)
        if len(anno) > 0:
            keypoint_info = male20['keypoint_info']
            skeleton_info = male20['skeleton_info']
            keypoints = np.asarray(multipleKpts)
            scores = np.asarray(multipleScores)
            if len(keypoints.shape) == 2:
                keypoints = keypoints[None, :, :]
                scores = scores[None, :, :]

            num_instance = keypoints.shape[0]
            for i in range(num_instance):
                opencv_image = draw_mmpose(opencv_image, keypoints[i], scores[i], keypoint_info,
                                           skeleton_info, 0.5, base_line_width*3, base_line_width)
        pil_image = Image.fromarray(
            cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
        return pil_image

    def drawBBoxes(self, image, anno, threshold=0.5):

        draw = ImageDraw.Draw(image)
        w_orig, h_orig = image.size

        # 根据图像大小调整边框和字体大小
        base_font_size = max(12, max(w_orig, h_orig) // 20)  # 最小字体为12
        base_line_width = max(1, max(w_orig, h_orig) // 200)  # 最小线宽为1
        font = self.load_font(base_font_size)

        id_to_class = {int(k): v for k, v in anno['CLS'].items()}
        color_dict = self.generate_color_dict(id_to_class)
        for bbox in anno['RESULTS']:
            # 遍历每个bbox
            confidence, class_id, x, y, w, h = bbox
            if confidence < threshold:
                continue

            class_id = int(class_id)
            # 将归一化的坐标转换为实际像素坐标
            x_min = int((x - w / 2) * w_orig)
            y_min = int((y - h / 2) * h_orig)
            x_max = int((x + w / 2) * w_orig)
            y_max = int((y + h / 2) * h_orig)

            # 获取类别名称和对应颜色
            class_name = id_to_class.get(class_id, "Unknown")
            color = color_dict.get(class_id, (255, 0, 0))  # 默认颜色为红色

            # 画出矩形框
            draw.rectangle([x_min, y_min, x_max, y_max],
                           outline=color,
                           width=base_line_width)

            # 构建标签
            label = f"{class_name} ({confidence:.2f})"
            text_bbox = draw.textbbox((x_min, y_min), label, font=font)
            text_background = [
                text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2,
                text_bbox[3] + 2
            ]

            # 绘制标签黑色背景和带颜色的文本
            draw.rectangle(text_background, fill="black")
            draw.text((x_min, y_min), label, fill=color, font=font)

        return image

    def genTestResult(self):
        preparedImages = []
        with open(self.dsPath, 'r') as f:
            self.imageInfoList = json.load(f)
        self.imagesWeightList = np.asarray(
            [imageInfo['WEIGHT'] for imageInfo in self.imageInfoList])
        self.imagesWeightList /= self.imagesWeightList.sum()
        self.resample_ds_by_weight()

        col = 5
        row = 5
        n = col * row

        for i, idxList in enumerate([
                self.imagesIdxList[i:i + n]
                for i in range(0, len(self.imagesIdxList), n)
        ]):
            preparedImages = []
            for idx in idxList:
                dsDir = os.path.dirname(self.dsPath)
                img = Image.open(os.path.join(
                    dsDir, self.imageInfoList[idx]['IMG'])).convert('RGB')

                img = self.drawBBoxes(img, self.imageInfoList[idx]['OBJS'])
                img = self.drawSkleton(
                    img, self.generateKeyPoint(self.imageInfoList[idx]))
                img = PIL.ImageOps.pad(img, (512, 512))
                preparedImages.append(img)
            self.genImageJiasaw(preparedImages, 512, 512, col, row, os.path.join(
                'TestResultSample', f'{i}.webp'))


wt = PoseDsCreator('xxxx')
wt.genTestResult()
