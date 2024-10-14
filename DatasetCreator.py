import json
import argparse
import pathlib
import pickle
import random
from shutil import copyfile
import os
from math import sqrt
import joblib
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import webdataset as wds
from PIL import Image
import io
import cv2
import tarfile
from MultiDatasetExtractor import MultiDatasetExtractor
import polars as pl
import numpy as np

class ImageDsCreator:

    def __init__(self, topDir, outputDir, filteredDirList) -> None:
        self.topDir = topDir
        self.multiDsExtractor = MultiDatasetExtractor(topDir)
        self.multiDsExtractor.scanDir(printScanResult=True)
        self.outputDir = outputDir
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)
        self.imageInfoListList = []
        self.imageSetList = []
        self.filteredDirList = filteredDirList

    def isInOrIsFilterDir(self, dir, filteredDirList):

        filteredDirList = [
            pathlib.Path(dirPath) for dirPath in filteredDirList
        ]
        
        dir = pathlib.Path(dir)
        if dir.is_absolute():
            dirRelativepath = dir.relative_to(self.topDir)
        else:
            dirRelativepath = dir

        detectedFilterDir = False
        for filterd in filteredDirList:
            if filterd in dirRelativepath.parents or filterd==dirRelativepath:
                detectedFilterDir = True
                break
        return detectedFilterDir

    def generateCandidateList(self, criteria, wantNum):
        for path in self.multiDsExtractor.dirsHasImageInfoJson:
            if not self.isInOrIsFilterDir(os.path.dirname(path), self.filteredDirList):
                self.addImageSet(path, criteria, wantNum)
            else:
                print(f'Skip {path}')

    def filterImageInfoList(self):
        totalImageNum = 0
        for jsonPath, criteria, wantedNum in self.imageSetList:
            with open(jsonPath, 'r') as f:
                imageInfo = json.load(f)

            relDsDir = pathlib.Path(jsonPath).parent.relative_to(self.topDir)
            
            filterList = [
                singleImageInfo for singleImageInfo in imageInfo
                if criteria(singleImageInfo) and \
                    not self.isInOrIsFilterDir(relDsDir/singleImageInfo['IMG'], self.filteredDirList)
            ]
            print('Choose from: %s' % jsonPath)
            if len(filterList) < wantedNum:
                print('Wanted num > Chosen num!')
            else:
                filterList = random.sample(filterList, wantedNum)
            numChosen = len(filterList)
            numTotal = len(imageInfo)
            print('  Chosen: %d\n  Total: %d\n  Chosen proportion:%.3f' %
                  (numChosen, numTotal, numChosen / numTotal))

            totalImageNum = totalImageNum + len(filterList)

            self.imageInfoListList.append((str(relDsDir.as_posix()), filterList))

        print('Total image num after filtering: %s' % totalImageNum)

    def balanceDatasetByPose(self, posePklDir):
        clusterModel = joblib.load(os.path.join(posePklDir,
                                                'ClusterModel.pkl'))
        with open(os.path.join(posePklDir, 'PoseImagesList.pkl'), 'rb') as f:
            poseImagesList = pickle.load(f)

        poseImagesDF = pl.DataFrame({
            'IMG': poseImagesList,
            'LBL': clusterModel.labels_
        })

        filterImagesInfoList = []
        for imageRelRoot, imageInfoList in self.imageInfoListList:
            for imageInfo in imageInfoList:
                imageInfo['IMG'] = os.path.join(imageRelRoot, imageInfo['IMG'])
                filterImagesInfoList.append(imageInfo)

        filterImagesDF = pl.DataFrame(data=filterImagesInfoList)
        filterImagesWithLabelsDF = filterImagesDF.join(poseImagesDF,
                                                       on="IMG",
                                                       how="left",
                                                       coalesce=True).with_columns(pl.col("LBL").fill_null(-1))
        freqDF = filterImagesWithLabelsDF.group_by(
            "LBL").agg(pl.col("IMG").len().alias('freq'))

        freqWithPoseDF = freqDF.filter(pl.col('LBL') != -1)
        freqWithoutPoseDF = freqDF.filter(pl.col('LBL') == -1)


        imageWithPoseNum = freqWithPoseDF.select(pl.sum('freq'))[0, 0]
        imageWithOutPoseNum = freqWithoutPoseDF.select(pl.sum('freq'))[0, 0]

        maxFreq = freqWithPoseDF.select(pl.max('freq'))[0, 0]

        freqWithPoseValidDF = freqWithPoseDF.filter(pl.col('freq') >=maxFreq/10)
        freqWithPoseNoiseDF = freqWithPoseDF.filter(pl.col('freq') <maxFreq/10)

        minFreq = freqWithPoseValidDF.select(pl.min('freq'))[0, 0]


        freqWithPoseInvalidDF = pl.concat(
            [
                freqWithoutPoseDF,
                freqWithPoseNoiseDF,
            ],
            how="vertical",
        )

        normalizeFreqWithPoseDF = freqWithPoseValidDF.select(pl.col('LBL'),
                                                        ((1/pl.col('freq'))/(1/minFreq)*(pl.sum('freq')/(imageWithPoseNum+imageWithOutPoseNum))).alias('WEIGHT')).with_columns(LBL_NEW=pl.col('LBL'))
        freqWithPoseInvalidDF = freqWithPoseInvalidDF.select(pl.col('LBL'),
                                                              (pl.sum('freq')/(imageWithPoseNum+imageWithOutPoseNum)).alias('WEIGHT')).with_columns(LBL_NEW=pl.lit(-1))
        normalizeFreDF = pl.concat(
            [
                normalizeFreqWithPoseDF,
                freqWithPoseInvalidDF,
            ],
            how="vertical",
        )

        


        
        fianlDF = filterImagesWithLabelsDF.join(
            normalizeFreDF, on="LBL", how="left", coalesce=True)
        
        
        lblArray = fianlDF['LBL_NEW'].to_numpy().copy()
        lblList = np.unique(lblArray)
        lblNexIdcs = range(len(lblList))
        reMapLbl = {}
        for oldLbl,newLbl in zip(lblList,lblNexIdcs):
            reMapLbl[oldLbl] = newLbl
        n_lblArray = lblArray.copy()
        for i,lbl in enumerate(lblArray):
            n_lblArray[i]=reMapLbl[lbl]

        weightArray = fianlDF['WEIGHT'].to_numpy()
        
        poseLblBins = lblNexIdcs #normalizeFreqWithPoseDF['LBL_NEW'].to_numpy(allow_copy=True)
        poseLblBins = np.sort(poseLblBins)

        weightArray = weightArray/sum(weightArray)
        sampleResult = np.random.choice(n_lblArray,100000,p=weightArray)

        # sampleResult = torch.multinomial(torch.tensor(weightArray), 100000, replacement=True).numpy()
        # sampleResult = n_lblArray[sampleResult]

        fig = plt.figure(figsize=(100, 8))
        ax = fig.add_subplot(1,1,1)
        ax.set_title('PDF')
        ns, edgeBin, bars  = ax.hist(sampleResult, bins=poseLblBins, rwidth=0.8,label='LBL',log=True)
        plt.bar_label(bars)
        ax.legend(prop={'size': 10})
        plt.grid(True)
        plt.savefig('sample.jpg')

        fianlDF.write_json(os.path.join(self.outputDir,'ImageInfoWeighted.json'))


    def generate(self, flattenDir=False):
        for imageRoot, imageInfoList in self.imageInfoListList:
            print('Copy %d images to %s from %s' %
                  (len(imageInfoList), self.outputDir, imageRoot))
            for singleImageInfo in tqdm(imageInfoList):
                orinPath = os.path.join(imageRoot, singleImageInfo['IMG'])

                if flattenDir:
                    targetDir = self.outputDir
                    newFileName = singleImageInfo['IMG'].replace('/',
                                                                 '_').replace(
                                                                     '\\', '_')
                    singleImageInfo['IMG'] = newFileName
                    targetPath = os.path.join(self.outputDir, newFileName)
                else:
                    targetDir = os.path.join(
                        self.outputDir,
                        os.path.dirname(singleImageInfo['IMG']))
                    targetPath = os.path.join(self.outputDir,
                                              singleImageInfo['IMG'])
                if not os.path.isdir(targetDir):
                    os.makedirs(targetDir)
                copyfile(orinPath, targetPath)
                self.imageInfoList.append(singleImageInfo)
        with open(self.imageInfoFilePath, 'w') as f:
            json.dump(self.imageInfoList, f)

    def addImageSet(self, jsonPath, criteria, wantedNum):
        self.imageSetList.append((jsonPath, criteria, wantedNum))


def criteria1(singleImageInfo): return singleImageInfo['Q512'] > 50 and singleImageInfo['H'] * \
    singleImageInfo['W'] > 896*896 and singleImageInfo['A_EAT'] > 5.5


def criteria2(singleImageInfo): return singleImageInfo['Q512'] > 50 and singleImageInfo['H'] * \
    singleImageInfo['W'] > 896*896 and singleImageInfo['A'] > 2


def criteria3(singleImageInfo): return singleImageInfo['Q512'] > 10 and singleImageInfo['H'] * \
    singleImageInfo['W'] > 896*896 and singleImageInfo['A'] > 2


def criteria4(singleImageInfo): return singleImageInfo['Q512'] > 65 and singleImageInfo['H'] * \
    singleImageInfo['W'] >= 896*896 and singleImageInfo['A'] > 5.2


def criteriaQF(singleImageInfo): return singleImageInfo['Q512'] > 70 and singleImageInfo['QF'] > 80 and singleImageInfo['H'] * \
    singleImageInfo['W'] >= 896*896 and singleImageInfo['A'] > 5.2


def criteriaFinest(singleImageInfo): return singleImageInfo['Q512'] > 70 and singleImageInfo['A_EAT'] > 6


def criteriaSPAQ(singleImageInfo): return singleImageInfo['SPAQ'] > 65 and singleImageInfo['H'] * \
    singleImageInfo['W'] >= 1024*1024


if __name__ == '__main__':
    dsc = ImageDsCreator('xxxx',
                         'xxxx')
    dsc.generateCandidateList(criteria1, 500000, filterDirList=['yyy'])
    dsc.filterImageInfoList()
    dsc.balanceDatasetByPose('PoseClusterCache')
