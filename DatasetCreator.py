import json
import argparse
import pathlib
import pickle
import random
from shutil import copyfile
import os
from math import sqrt
import joblib
from tqdm import tqdm
import webdataset as wds
from PIL import Image
import io
import cv2
import tarfile
from MultiDatasetExtractor import MultiDatasetExtractor
import polars as pl


class ImageDsCreator:

    def __init__(self, topDir, outputDir) -> None:
        self.topDir = topDir
        self.multiDsExtractor = MultiDatasetExtractor(topDir)
        self.multiDsExtractor.scanDir(printScanResult=True)
        self.outputDir = outputDir
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)
        self.imageInfoListList = []
        self.imageSetList = []

    def isInFilterDir(self, dir, filteredDirList):

        filteredDirList = [
            pathlib.Path(dirPath) for dirPath in filteredDirList
        ]
        dirRelativepath = pathlib.Path(os.path.relpath(dir, self.topDir))

        detectedFilterDir = False
        for filterd in filteredDirList:
            if filterd in dirRelativepath.parents:
                detectedFilterDir = True
                break
        return detectedFilterDir

    def generateCandidateList(self, criteria, wantNum, filterDirList=[]):
        for path in self.multiDsExtractor.dirsHasImageInfoJson:
            if not self.isInFilterDir(os.path.dirname(path), filterDirList):
                self.addImageSet(path, criteria, wantNum)
            else:
                print(f'Skip {path}')

    def filterImageInfoList(self):
        totalImageNum = 0
        for jsonPath, criteria, wantedNum in self.imageSetList:
            with open(jsonPath, 'r') as f:
                imageInfo = json.load(f)
            filterList = [
                singleImageInfo for singleImageInfo in imageInfo
                if criteria(singleImageInfo)
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
            relDsDir = str(
                pathlib.Path(
                    os.path.relpath(os.path.dirname(jsonPath),
                                    self.topDir)).as_posix())
            self.imageInfoListList.append((relDsDir, filterList))

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
        minFreq = freqWithPoseDF.select(pl.min('freq'))[0, 0]

        normalizeFreqWithPoseDF = freqWithPoseDF.select(pl.col('LBL'),
                                                        ((1/pl.col('freq'))/(1/minFreq)*(imageWithPoseNum/(imageWithPoseNum+imageWithOutPoseNum))).alias('WEIGHT'))
        normalizeFreqWithoutPoseDF = freqWithoutPoseDF.select(pl.col('LBL'),
                                                              (pl.col('freq')/(imageWithPoseNum+imageWithOutPoseNum)).alias('WEIGHT'))

        normalizeFreDF = pl.concat(
            [
                normalizeFreqWithPoseDF,
                normalizeFreqWithoutPoseDF,
            ],
            how="vertical",
        )
        fianlDF = filterImagesWithLabelsDF.join(
            normalizeFreDF, on="LBL", how="left", coalesce=True)
        fianlDF.write_json(os.path.join(self.outputDir,'ImageInfoWeighted.json'))

    def generateWdsDataset(self):
        sink = wds.ShardWriter(os.path.join(self.outputDir,
                                            "FinalDsWds-%05d.tar"),
                               maxsize=100 * 1024 * 1024)
        for imageRoot, imageInfoList in self.imageInfoListList:
            print('Writing images from: %s' % imageRoot)
            for imageInfo in tqdm(imageInfoList):

                imageBytes, imgRelPath = self.processImage(
                    imageRoot, imageInfo)
                sample = {
                    "__key__": os.path.splitext(imgRelPath)[0],
                    os.path.splitext(imgRelPath)[1][1:]: imageBytes,
                    "json": json.dumps(imageInfo).encode()
                }
                sink.write(sample)
        sink.close()

    def generateTar(self, tarName='FinalDsWds', imageInfoFileOnly=False):
        tar = tarfile.open(os.path.join(self.outputDir, tarName + ".tar"),
                           "w:")
        for imageRoot, imageInfoList in self.imageInfoListList:
            print('Writing images from: %s' % imageRoot)
            for imageInfo in tqdm(imageInfoList):
                if not imageInfoFileOnly:
                    imageBytes, imgRelPath = self.processImage(
                        imageRoot, imageInfo, passthrough=True)
                    info = tarfile.TarInfo(name=tarName + '/' + imgRelPath)
                    info.size = len(imageBytes)
                    tar.addfile(info, fileobj=io.BytesIO(imageBytes))
                self.imageInfoList.append(imageInfo)

        imageInfoJsonBytes = json.dumps(self.imageInfoList).encode('utf8')
        info = tarfile.TarInfo(name=tarName + '/ImageInfo.json')
        info.size = len(imageInfoJsonBytes)
        tar.addfile(info, fileobj=io.BytesIO(imageInfoJsonBytes))
        tar.close()
        with open(self.imageInfoFilePath, 'w') as f:
            json.dump(self.imageInfoList, f)

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
