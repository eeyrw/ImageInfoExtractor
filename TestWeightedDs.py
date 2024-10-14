import json
import math
from time import time
import PIL
from matplotlib import pyplot as plt
import numpy as np

import os
import random
from PIL import Image, ImageFont, ImageDraw  
from pillow_heif import register_heif_opener

from MultiDatasetExtractor import MultiDatasetExtractor
register_heif_opener()


class WeightedDsTester:
    def __init__(self, dsPath) -> None:
        self.dsPath = dsPath

    def constantAreaResize(self,w,h,area):
        resize_ratio = math.sqrt(area/(w*h))
        return tuple(math.ceil(x * resize_ratio) for x in (w,h))


    def genImageJiasaw(self, imageList, width, height, col, row, outPath):
        to_image = Image.new('RGB', (col * width, row * height))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(row):
            for x in range(col):
                from_image = imageList[y*col+x]
                to_image.paste(from_image, (x * width, y * height))
        return to_image.save(outPath, quality=90)  # 保存新图
    

    def __len__(self) -> int:
        return len(self.imagesIdxList)
    
    def resample_ds_by_weight(self):
        print('Resample DS')
        rawImagesIdxList = list(range(len(self.imageInfoList)))
        rng = np.random.default_rng()
        resampleList = rng.choice(rawImagesIdxList,len(self.imageInfoList)*10,replace=True,p=self.imagesWeightList)
        self.imagesIdxList = resampleList.tolist()

    def genTestResult(self):
        preparedImages = []
        with open(self.dsPath, 'r') as f:
            self.imageInfoList = json.load(f)
        self.imagesWeightList = np.asarray([imageInfo['WEIGHT'] for imageInfo in self.imageInfoList])
        self.imagesWeightList /= self.imagesWeightList.sum()
        self.resample_ds_by_weight()

        occupyList = np.zeros(len(self.imageInfoList))
        x = []
        y = []
        for i,idx in enumerate(self.imagesIdxList,start=1):
            occupyList[idx] = 1
            if i%1000==0:
                #print(f'{i}: {sum(occupyList)/len(self.imageInfoList)}')
                x.append(i)
                y.append(sum(occupyList)/len(self.imageInfoList))

        plt.figure(figsize=(50, 24))
        plt.plot(x, y, marker='o', linestyle='-')


        # Add title and labels
        plt.title('Sampleing process')
        plt.xlabel('steps')
        plt.ylabel('sample cover')

        # Display grid
        plt.grid(True)
        plt.savefig('Sampleing process.jpg')
        plt.close()

        col = 10
        row = 10
        n = col*row

        for i,idxList in enumerate([self.imagesIdxList[i:i + n] for i in range(0, len(self.imagesIdxList), n)]):
            preparedImages = []
            for idx in idxList:
                dsDir = os.path.dirname(self.dsPath)
                img = Image.open(os.path.join(dsDir, self.imageInfoList[idx]['IMG']))
                img = PIL.ImageOps.pad(img,(256,256))
                preparedImages.append(img)
            self.genImageJiasaw(preparedImages,256,256,col,row,os.path.join('TestResultSample',f'{i}.webp'))


wt = WeightedDsTester('xxxx')
wt.genTestResult()