import sys
sys.path.append("./hyperIQA")
sys.path.append("./BLIP")
sys.path.append("./TorchDeepDanbooru")
import Aesthetic
import TorchDeepDanbooru.inference
from PIL import Image
import BLIP.predict_simple
from hyperIQA.inference import Predictor, pil_loader
import os
import json
import argparse
import pathlib
from tqdm import tqdm
from pillow_heif import register_heif_opener
register_heif_opener()


class ImageSizeInfoCorrectTool:
    def __init__(self, topDir) -> None:
        pass

    def update(self, imageInfo, topDir):
        with open(os.path.join(topDir, imageInfo['IMG']), 'rb') as f:
            img = Image.open(f)
        width, height = img.size
        if 'W' in imageInfo.keys() and 'H' in imageInfo.keys():
            if width != imageInfo['W'] or height != imageInfo['H']:
                print('Correct size: %s' % imageInfo['IMG'])
                imageInfo.update({'W': width, 'H': height})
        else:
            print('Create size: %s' % imageInfo['IMG'])
            imageInfo.update({'W': width, 'H': height})
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['H', 'W'])


class ImageQuailityTool:
    def __init__(self, topDir) -> None:
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
        return set(['Q512', 'H', 'W'])


class ImageAestheticTool:
    def __init__(self, topDir) -> None:
        self.imageAestheticPredictor = Aesthetic.Predictor()

    def update(self, imageInfo, topDir):
        img = pil_loader(os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageAestheticPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['A', 'H', 'W'])


class ImageCaptionTool:
    def __init__(self, topDir, captionModel='BLIP') -> None:
        captionFile = os.path.join(topDir, 'CustomCaptionPool.txt')
        if os.path.isfile(captionFile):
            customCaptionPool = []
            with open(captionFile) as f:
                for line in f:
                    customCaptionPool.append(line.strip())
            print('Use custom caption: %s' % customCaptionPool)
        else:
            customCaptionPool = None
        if captionModel == 'BLIP':
            self.imageCaptionPredictor = BLIP.predict_simple.Predictor(
                customCaptionPool=customCaptionPool)
        elif captionModel == 'DeepDanbooru':
            self.imageCaptionPredictor = TorchDeepDanbooru.inference.Predictor()

    def update(self, imageInfo, topDir):
        if imageInfo['Q512'] > 60:
            img = pil_loader(os.path.join(topDir, imageInfo['IMG']))
            captionDictList = self.imageCaptionPredictor.predict(img)
            imageInfo.update({'CAP': [captionDict['caption']
                                      for captionDict in captionDictList]})
            havePrintFileName = False
            for captionDict in captionDictList:
                if captionDict['isCustomCap']:
                    if not havePrintFileName:
                        print('File:'+imageInfo['IMG'])
                        havePrintFileName = True
                    print('Custom cap: rank %s cap %s' %
                          (captionDict['rank'], captionDict['caption']))
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['CAP'])


class ImageInfoManager:
    def __init__(self, topDir, imageInfoFileName='ImageInfo.json', processTools=[]) -> None:
        self.topDir = topDir
        self.processTools = processTools
        self.imageInfoFilePath = os.path.join(self.topDir, imageInfoFileName)
        self.supportImageFormatList = ['.jpg','.webp','.png','.heic']
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
                if ext in self.supportImageFormatList:
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
        print('%s images found.' % len(imageList))
        return imageList

    def infoUpdate(self):
        processToolNameListDict = {}
        for processTool in self.processTools:
            processToolClass = processTool['toolClass']
            processToolNameListDict[processToolClass] = {
                'fieldSet': processToolClass.fieldSet(), 'forceUpdate': processTool['forceUpdate'], 'itemIdx': []}

        for idx, imageInfo in enumerate(self.imageInfoList):
            for processTool, processDict in processToolNameListDict.items():
                if processDict['forceUpdate'] or len(processDict['fieldSet']-set(imageInfo.keys())) > 0:
                    processDict['itemIdx'].append(idx)

        for processTool, processDict in processToolNameListDict.items():
            if len(processDict['itemIdx']) > 0:
                print('Tool: %s' % processTool.__name__)
                toolInstance = processTool(self.topDir)
                for i, imageInfoIdx in enumerate(tqdm(processDict['itemIdx'])):
                    try:
                        toolInstance.update(
                            self.imageInfoList[imageInfoIdx], self.topDir)
                    except Exception as e:
                        print('ERROR:%s:%s' %
                              (self.imageInfoList[imageInfoIdx], str(e)))

                    if i % 1000 == 0:
                        self.saveImageInfoList()
                self.saveImageInfoList()
            else:
                print('No update by %s' % processTool.__name__)
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
    parser.add_argument('--dsDir', type=str,
                        default=r"F:\NSFW_DS\sj_HEIC")
    config = parser.parse_args()
    tools = [{'toolClass': ImageSizeInfoCorrectTool, 'forceUpdate': False},
             {'toolClass': ImageQuailityTool, 'forceUpdate': False},
             {'toolClass': ImageAestheticTool, 'forceUpdate': False},
             {'toolClass': ImageCaptionTool, 'forceUpdate': False}]
    imageInfoManager = ImageInfoManager(config.dsDir, processTools=tools)
    imageInfoManager.updateImages()
    imageInfoManager.infoUpdate()
    imageInfoManager.saveImageInfoList()
