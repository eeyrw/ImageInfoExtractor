# Authors: Fabian Pedregosa <fabian.pedregosa@inria.fr>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Gael Varoquaux
# License: BSD 3 clause (C) INRIA 2011
import json
import math
import pathlib
import pickle
from time import time
import PIL
import joblib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from matplotlib import offsetbox
from rtmlib import draw_skeleton
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, mixture, random_projection, neighbors)
import os
import random
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from PIL import Image, ImageFont, ImageDraw  
from pillow_heif import register_heif_opener
from sklearn.metrics import silhouette_score

from MultiDatasetExtractor import MultiDatasetExtractor
register_heif_opener()


class PoseAnalyser:

    def __init__(self, dir, cacheDir='PoseClusterCache') -> None:
        self.multiDsExtractor = MultiDatasetExtractor(dir)
        self.multiDsExtractor.scanDir(printScanResult=True)
        self.visualSubsetIndcs=None
        self.poseEmbeddingList=None
        self.poseLabels=None
        self.poseImagesList=None
        self.rawPoseDict = None
        self.topDir = dir
        self.cacheDir = cacheDir

    def constantAreaResize(self,w,h,area):
        resize_ratio = math.sqrt(area/(w*h))
        return tuple(math.ceil(x * resize_ratio) for x in (w,h))

    def genEmbedding(self):
        self.embeddingDict = {}
        self.rawPoseDict = {}
        kptNose = 0
        kptLShoulder = 5
        kptRShoulder = 6
        kptLHip = 11
        kptRHip = 12

        numHasBasicKpt = 0
        numHasNotBasicKpt = 0
        for path in self.multiDsExtractor.dirsHasImageInfoJson:
            print('====Processing %s' % path)
            dsDir = os.path.dirname(path)
            relDsDir = str(pathlib.Path(
                os.path.relpath(dsDir, self.topDir)).as_posix())
            with open(path, 'r') as f:
                self.imageInfoList = json.load(f)
            for imageInfo in self.imageInfoList:
                if 'POSE_KPTS' in imageInfo.keys() and len(imageInfo['POSE_KPTS']) > 0:
                    bboxAreaList=[]
                    for poseDict in imageInfo['POSE_KPTS']:
                        x1, y1, x2, y2 = poseDict['BBOX']
                        area = (x2-x1)*(y2-y1)
                        bboxAreaList.append(area)
                    areaOrdinalIdcs = np.argsort(bboxAreaList)
                    maxBBOXIdx = areaOrdinalIdcs[-1]

                    invalidIdxNum = 0
                    for idx in imageInfo['POSE_KPTS'][maxBBOXIdx]['INVLD_KPTS_IDX']:
                        if idx<17:
                            invalidIdxNum=invalidIdxNum+1
                    
                    if invalidIdxNum>17-3:
                        continue

                    x1, y1, x2, y2 = imageInfo['POSE_KPTS'][maxBBOXIdx]['BBOX']
                    w = x2-x1
                    h = y2-y1
                    #idx = np.r_[0:17, 91:133]
                    idx = np.r_[0:17]
                    #idx = np.r_[0:133]
                    #idx = np.r_[5:17]
                    x_coords = np.asarray(imageInfo['POSE_KPTS'][maxBBOXIdx]['KPTS_X'])
                    y_coords = np.asarray(imageInfo['POSE_KPTS'][maxBBOXIdx]['KPTS_Y'])
                    nopIdx = imageInfo['POSE_KPTS'][maxBBOXIdx]['INVLD_KPTS_IDX']
                    resize_ratio = math.sqrt(1/(w*h))
                    x_coords = (x_coords-x1)*resize_ratio
                    y_coords = (y_coords-y1)*resize_ratio

                    if kptNose not in nopIdx and kptLShoulder not in nopIdx and kptRShoulder not in nopIdx:
                        numHasBasicKpt = numHasBasicKpt+1
                        x_center = np.mean(x_coords[[kptNose,kptLShoulder,kptRShoulder]])
                        y_center = np.mean(y_coords[[kptNose,kptLShoulder,kptRShoulder]])
                    elif kptLHip not in nopIdx and kptRHip not in nopIdx:
                        numHasBasicKpt = numHasBasicKpt+1
                        x_center = np.mean(x_coords[[kptLHip,kptRHip]])
                        y_center = np.mean(y_coords[[kptLHip,kptRHip]])     
                    else:
                        x_center = 0.5
                        y_center = 0.5
                        numHasNotBasicKpt = numHasNotBasicKpt+1

                    # x_center = np.mean(x_coords)
                    # y_center = np.mean(y_coords)

                    x_coords = x_coords-x_center
                    y_coords = y_coords-y_center

                    x_coords[nopIdx] = 0.0
                    y_coords[nopIdx] = 0.0

                    self.embeddingDict[os.path.join(relDsDir,imageInfo['IMG'])
                                    ] = np.concatenate((x_coords[idx],y_coords[idx]))
                    self.rawPoseDict[os.path.join(relDsDir,imageInfo['IMG'])] = imageInfo['POSE_KPTS']

        print(f'Num has basic kpt:{numHasBasicKpt}, Num has not basic kpt:{numHasNotBasicKpt}')


    def cluster(self,clusterMethod = 'kmeans',clusterNum=500):
        print('Start clustering...')
        self.poseImagesList = list(self.embeddingDict.keys())
        self.poseEmbeddingList = np.concatenate([[self.embeddingDict[v]]
                                 for v in self.poseImagesList], axis=0)
        
        if clusterMethod == 'kmeans':
            self.clusterModel = KMeans(n_clusters=clusterNum, random_state=42, init='k-means++',n_init='auto',verbose=1,max_iter=10000,tol=1e-6).fit(self.poseEmbeddingList)
            self.poseLabels = self.clusterModel.labels_
        elif clusterMethod == 'gmm':
            self.clusterModel = mixture.GaussianMixture(
            n_components=clusterNum, covariance_type="full", max_iter=500,verbose=1,tol=1e-4).fit(self.poseEmbeddingList)
            self.poseLabels =  self.clusterModel.predict(self.poseEmbeddingList)

        self.cacheClusterResult()


        return self.poseLabels

    def clusterTrials(self,clusterMethod = 'kmeans'):
        print('Start clustering...')
        self.poseImagesList = list(self.embeddingDict.keys())
        self.poseEmbeddingList = np.concatenate([[self.embeddingDict[v]]
                                 for v in self.poseImagesList], axis=0)

        cNumList = []
        interiaList = []

        for clusterCenterNum in range(50,6000,50):
        
            if clusterMethod == 'kmeans':
                self.clusterModel = KMeans(n_clusters=clusterCenterNum, random_state=42, n_init='auto',verbose=1,max_iter=10000,tol=1e-5).fit(self.poseEmbeddingList)
                self.poseLabels = self.clusterModel.labels_
                cNumList.append(clusterCenterNum)
                interiaList.append(self.clusterModel.inertia_)
            elif clusterMethod == 'gmm':
                self.clusterModel = mixture.GaussianMixture(
                n_components=clusterCenterNum, covariance_type="full", max_iter=100,verbose=1,tol=1e-4).fit(self.poseEmbeddingList)
                self.poseLabels =  self.clusterModel.predict(self.poseEmbeddingList)

        plt.figure(figsize=(50, 24))
        plt.plot(cNumList, interiaList, marker='o', linestyle='-')


        # Add title and labels
        plt.title('Line Chart')
        plt.xlabel('cNumList')
        plt.ylabel('interiaList')

        # Display grid
        plt.grid(True)
        plt.savefig('kmeans_interia.jpg')
        plt.close()
        print(cNumList)
        print(interiaList)

        return self.poseLabels

    def cacheClusterResult(self):
        if not os.path.exists(self.cacheDir):
            os.mkdir(self.cacheDir)
        joblib.dump(self.clusterModel, os.path.join(self.cacheDir,'ClusterModel.pkl'))
        joblib.dump(self.clusterModel, os.path.join(self.cacheDir,'PoseEmbeddingList.pkl'))
        with open(os.path.join(self.cacheDir,'PoseImagesList.pkl'),'wb') as f:
            pickle.dump(self.poseImagesList,f)
        with open(os.path.join(self.cacheDir,'RawPoseDict.pkl'),'wb') as f:
            pickle.dump(self.rawPoseDict,f)

    def reduceDim(self):
        print("Computing t-SNE embedding")

        rng1 = np.random.default_rng()
        # 采样一部分向量可视化
        self.visualSubsetIndcs = rng1.choice(len(self.poseImagesList), min(2000, len(self.poseImagesList)))
        image_arr = np.concatenate([[self.embeddingDict[self.poseImagesList[i]]]
                                    for i in self.visualSubsetIndcs], axis=0)

        tsne = manifold.TSNE(n_components=2, init='pca',
                             random_state=0, n_jobs=32, perplexity=50,verbose=1)

        X_tsne = tsne.fit_transform(image_arr)

        return X_tsne
    
    def loadClusterResult(self):
        try:
            self.clusterModel = joblib.load(os.path.join(self.cacheDir,'ClusterModel.pkl'))
            self.poseEmbeddingList = joblib.load(os.path.join(self.cacheDir,'PoseEmbeddingList.pkl'))
            with open(os.path.join(self.cacheDir,'PoseImagesList.pkl'),'rb') as f:
                self.poseImagesList = pickle.load(f)
            with open(os.path.join(self.cacheDir,'RawPoseDict.pkl'),'rb') as f:
                self.rawPoseDict = pickle.load(f)
            return True
        except:
            return False
    def genImageJiasaw(self, imageList, width, height, col, row, outPath):
        to_image = Image.new('RGB', (col * width, row * height))  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for y in range(row):
            for x in range(col):
                from_image = imageList[y*col+x]
                to_image.paste(from_image, (x * width, y * height))
        return to_image.save(outPath, quality=90)  # 保存新图
    
    def analyzerCluster(self):
        self.clusterMap = {}

        for imageKey,lbl in zip(self.poseImagesList,self.clusterModel.labels_):
            lbl = lbl.item()
            if lbl in self.clusterMap.keys():
                self.clusterMap[lbl].append(imageKey)
            else:
                self.clusterMap[lbl] = [imageKey]
            
        
        clusterNumList= []

        for lbl,clusterItems in self.clusterMap.items():
            clusterNumList.append((lbl,len(clusterItems)))

        clusterNumListSorted = sorted(clusterNumList, key=lambda x:x[1])
        visualNum = 64

        if not os.path.exists('ClusterResultSample'):
            os.mkdir('ClusterResultSample')

        for lbl,clusterItems in self.clusterMap.items():
            if len(clusterItems)<visualNum:
                selectedSamples = clusterItems
            else:
                selectedSamples = random.sample(clusterItems,k=visualNum)
            preparedImages = []
            numSamples = len(selectedSamples)
            for sample in selectedSamples:
                img = Image.open(os.path.join(self.topDir, sample))
                img = PIL.ImageOps.pad(img,(256,256))
                preparedImages.append(img)

            if len(selectedSamples)<visualNum:
                col = len(selectedSamples)
                row = 1
            else:
                col = numSamples//8
                row = 8
            self.genImageJiasaw(preparedImages,256,256,col,row,os.path.join('ClusterResultSample',f'{len(clusterItems)}-{lbl}.webp'))



        print(clusterNumListSorted)



    def analyze(self):
        if not self.loadClusterResult():
            self.genEmbedding()
            self.cluster(clusterNum=888)
            if self.embeddingDict:
                X_tsne = self.reduceDim()
                self.visualizeClusterResult(X_tsne, self.poseLabels,
                                    "t-SNE embedding of the Pose")
        self.analyzerCluster()

    def analyzeClusterNum(self):
        self.genEmbedding()
        self.clusterTrials()

    def visualizeClusterResult(self, X, labels, title=None):

        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(f'{n_clusters_} clusters, {n_noise_} noise')

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        # matplotlib.rcParams['agg.path.chunksize'] = 10000
        # matplotlib.rcParams.update(matplotlib.rc_params())
        plt.figure(figsize=(150, 150))
        ax = plt.subplot(111)

        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 1e-6:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                img = Image.open(os.path.join(
                    self.topDir, self.poseImagesList[self.visualSubsetIndcs[i]])).convert('RGB')

                w,h = img.size
                img = img.resize(self.constantAreaResize(w,h,128*128))
                if labels[i] != -1:
                    color = colors[labels[self.visualSubsetIndcs[i]]]
                    color_pil = (
                        int(color[0] * 255),
                        int(color[1] * 255),
                        int(color[2] * 255),
                    )
                else:
                    color_pil = (0, 0, 0)
                img = PIL.ImageOps.expand(img, border=10, fill=color_pil)
                draw = ImageDraw.Draw(img)  
                draw.text((15, 15), str(labels[self.visualSubsetIndcs[i]]), align ="left",fill=(0,0,0))  
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(img), X[i], frameon=False)
                ax.add_artist(imagebox)
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)
        plt.savefig(title+'.jpg')
        # plt.show()
        plt.close()



# pa = PoseAnalyser('xxxx')
# pa.analyze()
#pa.analyzeClusterNum()
