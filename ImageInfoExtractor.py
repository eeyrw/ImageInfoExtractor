import sys
from pillow_heif import register_heif_opener
from tqdm import tqdm
import pathlib
import argparse
import json
import os
import yaml
import hpyerIQAInference.inference
import FBCNNInference.inference
import BLIPInference.predict_simple
from PIL import Image
import TorchDeepDanbooruInference.inference
import Aesthetic
import RealESRGANInference.inference_realesrgan
import RealCUGANInference.inference_cugan
# import ViTPoseInference.inference_vitpose_by_easypose
import SPAQInference.inference_SPAQ
import EATInference.inference
import BLIP2Inference.inference
import LlavaInference.inference
from shutil import copyfile, move
import open_clip
import math
import OCRInference.inference
import WatermarkDetectionInference.inference_simple
from PIL import ImageDraw
from pathlib import Path, PurePath
register_heif_opener()


class AdditionalMetaInfo:
    def __init__(self, topDir) -> None:
        with open(os.path.join(topDir, 'MetaInfo.json'), 'r') as f:
            self.metaInfo = json.load(f)

    def update(self, imageInfo, topDir):
        imageInfo.update(self.metaInfo)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set([])


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
        self.imageQualityPredictor = hpyerIQAInference.inference.Predictor(
            weightsDir='./DLToolWeights/HyperIQA')

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict_multiscale(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['Q512', 'H', 'W'])


class ImageSPAQTool:
    def __init__(self, topDir) -> None:
        self.imageQualityPredictor = SPAQInference.inference_SPAQ.Predictor(
            weightsDir='./DLToolWeights/SPAQ')

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['SPAQ', 'H', 'W'])


class WatermarkDetectTool:
    def __init__(self, topDir) -> None:
        self.watermarkPredictor = WatermarkDetectionInference.inference_simple.Predictor(
            weightsDir="./DLToolWeights/WatermarkDetection")

    def update(self, imageInfo, topDir):
        img = WatermarkDetectionInference.inference_simple.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))

        watermarkResult = self.watermarkPredictor.predict(img)

        # bakDir = os.path.join(topDir, 'watermark_result',
        #                       os.path.dirname(imageInfo['IMG']))
        # bakImagePath = os.path.join(
        #     bakDir, os.path.basename(imageInfo['IMG']))
        # if not os.path.exists(bakDir):
        #     os.makedirs(bakDir)
        # if watermarkResult['HAS_WATERMARK']>0.7:
        #     img.save(bakImagePath)

        imageInfo.update(watermarkResult)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['HAS_WATERMARK'])


class ImageOCRTool:
    def __init__(self, topDir) -> None:
        self.imageOCRPredictor = OCRInference.inference.Predictor(
            weightsDir="./DLToolWeights/EasyOCR")

    def update(self, imageInfo, topDir):
        img = OCRInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        resize_ratio = math.sqrt(1024*1024/(img.size[0]*img.size[1]))
        poseResult = img.resize(
            tuple(math.ceil(x * resize_ratio) for x in img.size),
            Image.BICUBIC
        )

        width, height = img.size
        bounds = self.imageOCRPredictor.predict(poseResult)

        bakDir = os.path.join(topDir, 'ocr_result',
                              os.path.dirname(imageInfo['IMG']))
        bakImagePath = os.path.join(
            bakDir, os.path.basename(imageInfo['IMG']))
        if not os.path.exists(bakDir):
            os.makedirs(bakDir)
        if len(bounds) > 0:
            self.draw_boxes(poseResult, bounds)
            poseResult.save(bakImagePath)

        imageInfo.update({'W': width, 'H': height})
        return imageInfo

    def draw_boxes(self, image, bounds, color='yellow', width=6):
        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        return image

    @staticmethod
    def fieldSet():
        return set(['H', 'W'])


class JpegQuailityTool:
    def __init__(self, topDir) -> None:
        self.imageQualityPredictor = FBCNNInference.inference.Predictor(
            weightsDir='./DLToolWeights/FBCNN')

    def update(self, imageInfo, topDir):
        img = FBCNNInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['QF', 'H', 'W'])


class ImageAestheticTool:
    def __init__(self, topDir) -> None:
        self.imageAestheticPredictor = Aesthetic.Predictor(
            weightsDir='./DLToolWeights/Aesthetic')

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageAestheticPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['A', 'H', 'W'])


class ImageEATAestheticTool:
    def __init__(self, topDir) -> None:
        self.imageAestheticPredictor = EATInference.inference.Predictor(
            weightsDir='./DLToolWeights/EAT')

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageAestheticPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['A_EAT', 'H', 'W'])


class ImageSRTool:
    def __init__(self, topDir) -> None:
        self.imageSRPredictor = RealESRGANInference.inference_realesrgan.Predictor(
            weightsDir='./DLToolWeights/RealESRGAN')

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        # if width*height < 768*768 and width*height > 384*384 and imageInfo['Q512'] > 60:
        # if width*height >1024*1024:
        #     resize_ratio = math.sqrt(1024*1024/(img.size[0]*img.size[1]))

        #     img = img.resize(
        #                 tuple(math.ceil(x * resize_ratio) for x in img.size),
        #                 Image.BICUBIC
        #             )

        srImg = self.imageSRPredictor.predict(img)
        bakDir = os.path.join(topDir, 'raw_before_sr',
                              os.path.dirname(imageInfo['IMG']))
        rawImagePath = os.path.join(topDir, imageInfo['IMG'])
        bakImagePath = os.path.join(
            bakDir, os.path.basename(imageInfo['IMG']))
        if not os.path.exists(bakDir):
            os.makedirs(bakDir)
        copyfile(rawImagePath, bakImagePath)
        savedPath = rawImagePath
        srImg.save(savedPath)
        width, height = srImg.size

        imageInfo.update({'W': width, 'H': height})
        return imageInfo

    @staticmethod
    def fieldSet():
        return set(['H', 'W'])

    @staticmethod
    def updateCriteria(imageInfo):
        imageArea = imageInfo['W']*imageInfo['H']
        return imageArea < 1024*1024 and imageArea > 384*384 and imageInfo['Q512'] > 60


# class ImagePoseEstimateTool:
#     def __init__(self, topDir) -> None:
#         self.imagePoseEstPredictor = ViTPoseInference.inference_vitpose_by_easypose.Predictor(
#             weightsDir='./DLToolWeights/EasyPose')

#     def update(self, imageInfo, topDir):
#         img = hpyerIQAInference.inference.pil_loader(
#             os.path.join(topDir, imageInfo['IMG']))

#         poseResult,preds = self.imagePoseEstPredictor.predict(img)
#         width, height = poseResult.size
#         if width*height >1024*1024:
#             resize_ratio = math.sqrt(1024*1024/(img.size[0]*img.size[1]))
#             poseResult = poseResult.resize(
#                         tuple(math.ceil(x * resize_ratio) for x in img.size),
#                         Image.BICUBIC
#                     )
#         bakDir = os.path.join(topDir, 'pose_est_result',
#                               os.path.dirname(imageInfo['IMG']))
#         bakImagePath = os.path.join(
#             bakDir, os.path.basename(imageInfo['IMG']))
#         if not os.path.exists(bakDir):
#             os.makedirs(bakDir)
#         poseResult.save(bakImagePath)
#         #imageInfo.update(preds)
#         return imageInfo

#     @staticmethod
#     def fieldSet():
#         return set(['H', 'W'])

class ImageFilterTool:
    def __init__(self, topDir) -> None:
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k',
                                                                               cache_dir='./DLToolWeights/OpenCLIP')
        self.tokenizer = open_clip.get_tokenizer(
            'ViT-H-14')
        self.model.to('cuda')

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))

        img = self.preprocess(img).unsqueeze(0).to('cuda')
        text = self.tokenizer(
            ["draft sketch", "finshed work"]).to('cuda')

        import torch
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(img)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @
                          text_features.T).softmax(dim=-1)[0]
            idx = torch.argmax(text_probs)
            if idx == 0:
                bakDir = os.path.join(topDir, 'raw_before_filter',
                                      os.path.dirname(imageInfo['IMG']))
                rawImagePath = os.path.join(topDir, imageInfo['IMG'])
                print('Filter out:%s' % rawImagePath)
                bakImagePath = os.path.join(
                    bakDir, os.path.basename(imageInfo['IMG']))
                if not os.path.exists(bakDir):
                    os.makedirs(bakDir)
                move(rawImagePath, bakImagePath)
            return imageInfo

    @staticmethod
    def fieldSet():
        return set([])


class ImageCaptionTool:
    def __init__(self, topDir, captionModel='LLAVA') -> None:
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
            self.imageCaptionPredictor = BLIPInference.predict_simple.Predictor(
                customCaptionPool=customCaptionPool, weightsDir='./DLToolWeights/BLIP')
        elif captionModel == 'DeepDanbooru':
            self.imageCaptionPredictor = TorchDeepDanbooruInference.inference.Predictor(
                weightsDir='./DLToolWeights/DeepDanbooru')
        elif captionModel == 'BLIP2':
            self.imageCaptionPredictor = BLIP2Inference.inference.Predictor(
                weightsDir='./DLToolWeights')
        elif captionModel == 'LLAVA':
            self.imageCaptionPredictor = LlavaInference.inference.Predictor(
                weightsDir='/large_tmp/')

    def update(self, imageInfo, topDir):
        if imageInfo['Q512'] > 35:
            img = hpyerIQAInference.inference.pil_loader(
                os.path.join(topDir, imageInfo['IMG']))
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
    def __init__(self, topDir, imageInfoFileName='ImageInfo.json', processTools=[], toolConfigYAML=None) -> None:
        self.topDir = topDir
        self.processTools = processTools
        self.toolConfigYAML = toolConfigYAML
        self.imageInfoFilePath = os.path.join(self.topDir, imageInfoFileName)
        self.supportImageFormatList = ['.jpg', '.webp', '.png', '.heic']
        if os.path.isfile(self.imageInfoFilePath):
            with open(self.imageInfoFilePath, 'r') as f:
                self.imageInfoList = json.load(f)
        else:
            print('ImageInfo File Not Found. Create one.')
            self.imageInfoList = []
        self.createProcessTools()

    def createProcessTools(self):
        if self.toolConfigYAML:
            print('Use tool config YAML,param processTools has been ignored.')
            with open(self.toolConfigYAML, 'r') as f:
                toolsConfig = yaml.safe_load(f)['Tools']
                toolsConfig = [] if toolsConfig is None else toolsConfig

            processTools = []
            for toolDict in toolsConfig:
                toolDictUpdate = {'forceUpdate': False, 'args': {}}
                toolDictUpdate.update(toolDict)
                toolDictUpdate['toolClass'] = getattr(
                    sys.modules[__name__], toolDict['toolClass'])
                processTools.append(toolDictUpdate)
            self.processTools = processTools

    def saveImageInfoList(self):
        with open(self.imageInfoFilePath, 'w') as f:
            json.dump(self.imageInfoList, f)

    def getImageList(self, filteredDirList=[], relPath=False):
        print('Detect image files...')
        imageList = []
        filteredDirList = [pathlib.Path(dirPath).as_posix()
                           for dirPath in filteredDirList]
        for root, dirs, files in os.walk(self.topDir):

            dirRelativepath = pathlib.Path(
                os.path.relpath(root, self.topDir)).as_posix()

            dirsAfterFilterd = []
            for d in dirs:
                detectedFilterDir = False
                for filterd in filteredDirList:
                    if PurePath(dirRelativepath+'/'+d) == PurePath(filterd):
                        detectedFilterDir = True
                        break
                if not detectedFilterDir:
                    dirsAfterFilterd.append(d)

            dirs[:] = dirsAfterFilterd

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

                    imageList.append(fullFilePath)
        print('%s images found.' % len(imageList))
        return imageList

    def infoUpdate(self):
        processToolNameListDict = {}
        for processTool in self.processTools:
            processToolClass = processTool['toolClass']
            processToolNameListDict[processToolClass] = {
                'fieldSet': processToolClass.fieldSet(),
                'forceUpdate': processTool['forceUpdate'],
                'args': processTool['args'],
                'itemIdx': []}

        for idx, imageInfo in enumerate(self.imageInfoList):
            for processTool, processDict in processToolNameListDict.items():
                meetUpdateCriteria = True
                if hasattr(processTool, 'updateCriteria'):
                    meetUpdateCriteria = processDict['forceUpdate'] or processTool.updateCriteria(
                        imageInfo)
                else:
                    meetUpdateCriteria = processDict['forceUpdate'] or len(
                        processDict['fieldSet']-set(imageInfo.keys())) > 0
                if meetUpdateCriteria:
                    processDict['itemIdx'].append(idx)

        for processTool, processDict in processToolNameListDict.items():
            if len(processDict['itemIdx']) > 0:
                print('Tool: %s' % processTool.__name__)
                toolInstance = processTool(self.topDir)
                for i, imageInfoIdx in enumerate(tqdm(processDict['itemIdx'])):
                    try:
                        toolInstance.update(
                            self.imageInfoList[imageInfoIdx], self.topDir, **processDict['args'])
                    except Exception as e:
                        raise e
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

        deletedImageItemsNum = len(deletedImageItems)
        if deletedImageItemsNum > 0:
            displayCounter = 10
            for delIdx in sorted([imageFileNameIndexDict[itemRelPath] for itemRelPath in deletedImageItems], reverse=True):
                if displayCounter > 0:
                    displayCounter = displayCounter - 1
                    print('Del:', self.imageInfoList[delIdx])
                elif displayCounter == 0:
                    print('Del: %d items to be displayed. Too much to show...' %
                          deletedImageItemsNum)
                    displayCounter = displayCounter - 1
                self.imageInfoList.pop(delIdx)

        newImageItemsNum = len(newImageItems)
        if newImageItemsNum > 0:
            displayCounter = 10
            for newImageRelPath in newImageItems:
                if displayCounter > 0:
                    displayCounter = displayCounter - 1
                    print('New:', {'IMG': newImageRelPath})
                elif displayCounter == 0:
                    print('New: %d items to be displayed. Too much to show...' %
                          newImageItemsNum)
                    displayCounter = displayCounter - 1

                self.imageInfoList.append({'IMG': newImageRelPath})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsDir', type=str,
                        default=r"your dataset dir")
    parser.add_argument('--multiDsDir', type=str,
                        default=None)
    parser.add_argument('--toolConfig', type=str,
                        default=None)
    config = parser.parse_args()
    config.toolConfig = 'MyExtractionPipeline.yaml'
    config.multiDsDir = False  # "True"
    if config.multiDsDir:
        with os.scandir(r'your dataset dir') as it:
            for entry in it:
                if entry.is_dir() and entry.name != 'original_images':
                    print(entry.path)

                    imageInfoManager = ImageInfoManager(
                        entry.path, toolConfigYAML=config.toolConfig)
                    imageInfoManager.updateImages(
                        filteredDirList=['raw_before_sr'])
                    imageInfoManager.infoUpdate()
                    imageInfoManager.saveImageInfoList()
    else:
        imageInfoManager = ImageInfoManager(
            config.dsDir, toolConfigYAML=config.toolConfig)
        imageInfoManager.updateImages(
            filteredDirList=['raw_before_sr', 'ocr_result',])
        imageInfoManager.infoUpdate()
        imageInfoManager.saveImageInfoList()
