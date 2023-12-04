import json
import argparse
import random
from shutil import copyfile
import os
from math import sqrt
from tqdm import tqdm
from PIL import Image
import io
import cv2
import tarfile


class ImageDsCreator:
    def __init__(self, outputDir) -> None:
        self.imageInfoList = []
        self.imageSetList = []
        self.outputDir = outputDir
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)
        self.imageInfoListList = []
        self.imageInfoFilePath = os.path.join(self.outputDir, 'ImageInfo.json')
        self.maxPixels = 1280*1280

    def filterImageInfoList(self):
        totalImageNum = 0
        for jsonPath, criteria, wantedNum in self.imageSetList:
            with open(jsonPath, 'r') as f:
                imageInfo = json.load(f)
            filterList = [
                singleImageInfo for singleImageInfo in imageInfo if criteria(singleImageInfo)]

            print('Images satisfy criteria:', len(filterList))

            if len(filterList) < wantedNum:
                print('Wanted num > selected num!')
            else:
                filterList = random.sample(filterList, wantedNum)
            totalImageNum = totalImageNum + len(filterList)
            self.imageInfoListList.append(
                (os.path.dirname(jsonPath), filterList))

        print('Total image num after filtering: %s' % totalImageNum)

    def processImage(self, imageRoot, imageInfo, convertToWebP=True, passthrough=False):
        relPath = imageInfo['IMG']
        with open(os.path.join(imageRoot, relPath), "rb") as stream:
            imageBytes = stream.read()
        if not passthrough:
            if convertToWebP or imageInfo['W']*imageInfo['H'] > self.maxPixels:
                outputImageBytesIO = io.BytesIO()
                with Image.open(io.BytesIO(imageBytes)) as im:
                    if imageInfo['W']*imageInfo['H'] > self.maxPixels:
                        actualW, actualH = im.size
                        aspectRatio = actualH/actualW
                        resizedW = sqrt(self.maxPixels/aspectRatio)
                        resizedH = resizedW*aspectRatio
                        resizedW = int(resizedW)
                        resizedH = int(resizedH)
                        raw_format = im.format
                        im = im.resize((resizedW, resizedH))
                        imageInfo['W'] = resizedW
                        imageInfo['H'] = resizedH
                    if convertToWebP:
                        raw_format = 'WebP'
                        relPath = os.path.splitext(relPath)[0]+'.webp'
                        imageInfo['IMG'] = relPath
                    im.save(outputImageBytesIO, format=raw_format)
                imageBytes = outputImageBytesIO.getvalue()
        return imageBytes, relPath

    def generateTar(self, tarName='FinalDsWds', imageInfoFileOnly=False):
        tar = tarfile.open(os.path.join(
            self.outputDir, tarName+".tar"), "w:")
        for imageRoot, imageInfoList in self.imageInfoListList:
            print('Writing images from: %s' % imageRoot)
            for imageInfo in tqdm(imageInfoList):
                if not imageInfoFileOnly:
                    imageBytes, imgRelPath = self.processImage(
                        imageRoot, imageInfo, passthrough=True)
                    info = tarfile.TarInfo(name=tarName+'/'+imgRelPath)
                    info.size = len(imageBytes)
                    tar.addfile(info, fileobj=io.BytesIO(imageBytes))
                self.imageInfoList.append(imageInfo)

        imageInfoJsonBytes = json.dumps(self.imageInfoList).encode('utf8')
        info = tarfile.TarInfo(name=tarName+'/ImageInfo.json')
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
                    newFileName = singleImageInfo['IMG'].replace(
                        '/', '_').replace('\\', '_')
                    singleImageInfo['IMG'] = newFileName
                    targetPath = os.path.join(
                        self.outputDir, newFileName)
                else:
                    targetDir = os.path.join(
                        self.outputDir, os.path.dirname(singleImageInfo['IMG']))
                    targetPath = os.path.join(
                        self.outputDir, singleImageInfo['IMG'])
                if not os.path.isdir(targetDir):
                    os.makedirs(targetDir)
                copyfile(orinPath, targetPath)
                self.imageInfoList.append(singleImageInfo)
        with open(self.imageInfoFilePath, 'w') as f:
            json.dump(self.imageInfoList, f)

    def addImageSet(self, jsonPath, criteria, wantedNum):
        self.imageSetList.append((jsonPath, criteria, wantedNum))

    def exportImageInfoList(self, jsonName='ImageInfoSelected.json',useJsonl=False):
        for imageRoot, imageInfoList in self.imageInfoListList:
            relPathOfDir = os.path.relpath(imageRoot,self.outputDir)
            for singleImageInfo in imageInfoList:
                singleImageInfo['IMG'] = os.path.join(relPathOfDir,singleImageInfo['IMG'])
                self.imageInfoList.append(singleImageInfo)
            
        exportPath = os.path.join(self.outputDir, jsonName)
        if useJsonl:
            import jsonlines
            exportPath =  os.path.splitext(exportPath)[0]+'.jsonl'
            with jsonlines.open(exportPath, mode='w') as writer:
                writer.write(self.imageInfoList)
        else:
            with open(exportPath, 'w') as f:
                json.dump(self.imageInfoList, f)
