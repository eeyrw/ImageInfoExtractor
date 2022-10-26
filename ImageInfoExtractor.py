import os
import sys
import json
from shutil import copyfile
import argparse
import pathlib
from tqdm import tqdm
import webdataset as wds

class ImageQuailityTool:
    def __init__(self) -> None:
        sys.path.append("./hyperIQA")
        from hyperIQA.inference import Predictor, pil_loader
        self.imageQualityPredictor = Predictor(
            r'hyperIQA\pretrained\koniq_pretrained.pkl')

    def update(self, imageInfo, topDir):
        img = pil_loader(os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict_multiscale(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['Q512', 'Q1024', 'Q2048', 'H', 'W'])


class ImageCaptionTool:
    def __init__(self) -> None:
        sys.path.append("./BLIP")
        import BLIP.predict_simple
        self.imageCaptionPredictor = BLIP.predict_simple.Predictor()

    def update(self, imageInfo, topDir):
        img = pil_loader(os.path.join(topDir, imageInfo['IMG']))
        caption = self.imageCaptionPredictor.predict(img)[0]
        imageInfo.update({'CAP': caption})
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['CAP'])


class ImageInfoManager:
    def __init__(self, topDir, imageInfoFileName='ImageInfo.json', processTools=[]) -> None:
        self.topDir = topDir
        self.processTools = processTools
        self.imageInfoFilePath = os.path.join(self.topDir, imageInfoFileName)
        if os.path.isfile(self.imageInfoFilePath):
            with open(self.imageInfoFilePath, 'r') as f:
                self.imageInfoList = json.load(f)
        else:
            print('ImageInfo File Not Found. Create one.')
            self.imageInfoList = []

    def saveImageInfoList(self):
        with open(self.imageInfoFilePath, 'w') as f:
            json.dump(self.imageInfoList, f)

    def getImageList(self, filteredDirList=[], relPath=False):
        print('Detect image files...')
        imageList = []
        filteredDirList = [pathlib.Path(dirPath).as_posix()
                           for dirPath in filteredDirList]
        for root, dirs, files in os.walk(self.topDir):
            for filename in files:
                basename, ext = os.path.splitext(filename)
                ext = ext.lower()
                if ext == '.jpg':
                    if relPath:
                        fullFilePath = pathlib.Path(os.path.relpath(
                            os.path.join(root, filename), self.topDir)).as_posix()
                    else:
                        fullFilePath = pathlib.Path(
                            os.path.join(root, filename)).as_posix()
                    dirRelativepath = pathlib.Path(
                        os.path.relpath(root, self.topDir)).as_posix()
                    if dirRelativepath in filteredDirList:
                        continue
                    imageList.append(fullFilePath)
        print('%s images found.'% len(imageList))
        return imageList

    def infoUpdate(self):
        processToolNameListDict = {}
        for processTool in self.processTools:
            processToolNameListDict[processTool] = {
                'fieldSet': processTool.fieldSet(), 'itemIdx': []}

        for idx, imageInfo in enumerate(self.imageInfoList):
            for processTool, processDict in processToolNameListDict.items():
                if len(processDict['fieldSet']-set(imageInfo.keys())) > 0:
                    processDict['itemIdx'].append(idx)

        for processTool, processDict in processToolNameListDict.items():
            if len(processDict['itemIdx'])>0:
                print('Tool: %s' % processTool.__name__)
                toolInstance = processTool()
                for imageInfoIdx in tqdm(processDict['itemIdx']):
                    toolInstance.update(
                        self.imageInfoList[imageInfoIdx], self.topDir)
            else:
                print('No update by %s'%processTool.__name__)
                continue

    def updateImages(self, filteredDirList=[]):
        actualImageList = self.getImageList(filteredDirList, relPath=True)
        imageFileNameIndexDict = {
            imageInfo['IMG']: idx for idx, imageInfo in enumerate(self.imageInfoList)}
        orinImageInfoListPathSet = set(imageFileNameIndexDict.keys())

        actualImagePathSet = set(actualImageList)
        newImageItems = actualImagePathSet-orinImageInfoListPathSet
        deletedImageItems = orinImageInfoListPathSet-actualImagePathSet

        if len(deletedImageItems) > 0:
            displayCounter = 10
            for delIdx in sorted([imageFileNameIndexDict[itemRelPath] for itemRelPath in deletedImageItems], reverse=True):
                if displayCounter > 0:
                    displayCounter = displayCounter - 1
                    print('Del:', self.imageInfoList[delIdx])
                elif displayCounter == 0:
                    print('New: Too much to show...')
                    displayCounter = displayCounter - 1
                self.imageInfoList.pop(delIdx)

        if len(newImageItems) > 0:
            displayCounter = 10
            for newImageRelPath in newImageItems:
                if displayCounter > 0:
                    displayCounter = displayCounter - 1
                    print('New:', {'IMG': newImageRelPath})
                elif displayCounter == 0:
                    print('New: Too much to show...')
                    displayCounter = displayCounter - 1

                self.imageInfoList.append({'IMG': newImageRelPath})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsDir', type=str, default=r'C:\Users\uie38447\Desktop\Hardware Settings')
    config = parser.parse_args()
    tools = []#[ImageQuailityTool, ImageCaptionTool]
    imageInfoManager = ImageInfoManager(config.dsDir,processTools=tools)
    imageInfoManager.updateImages()
    imageInfoManager.infoUpdate()
    imageInfoManager.saveImageInfoList()
