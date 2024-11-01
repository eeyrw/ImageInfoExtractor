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
import WDVitTaggerV3.inference
import Aesthetic
import RealESRGANInference.inference_realesrgan
import RealCUGANInference.inference_cugan
import RTMPoseInference.inference
import SPAQInference.inference_SPAQ
import EATInference.inference
import BLIP2Inference.inference
import MiniCPMLlama3V25Inference.inference
import SmartCropInference.inference
from shutil import copyfile, move
# import open_clip
import math
import OCRInference.inference
import WatermarkDetectionInference.inference_simple
import YoloInference.inference
from PIL import ImageDraw
from pathlib import Path, PurePath
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import shutil
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

register_heif_opener()


class BatchInferenceDataset(Dataset):
    def __init__(self, topDir, imageInfoList, indexList, transform):
        self.transform = transform
        self.imageInfoList = imageInfoList
        self.indexList = indexList
        self.topDir = topDir

    def __len__(self):
        return len(self.indexList)

    def __getitem__(self, item):
        idx = self.indexList[item]
        image_path = os.path.join(self.topDir,
                                  self.imageInfoList[idx]['IMG'])
        image = Image.open(image_path).convert('RGB')
        x = self.transform(image)
        return idx, x


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
        ret = None
        if 'W' in imageInfo.keys() and 'H' in imageInfo.keys():
            if width != imageInfo['W'] or height != imageInfo['H']:
                print('Correct size: %s' % imageInfo['IMG'])
                imageInfo.update({'W': width, 'H': height})
                ret = imageInfo
        else:
            print('Create size: %s' % imageInfo['IMG'])
            imageInfo.update({'W': width, 'H': height})
            ret = imageInfo
        return ret

    @staticmethod
    def fieldSet():
        return set(['H', 'W'])


class ImageQuailityTool:
    def __init__(self, topDir,device='cuda') -> None:
        self.imageQualityPredictor = hpyerIQAInference.inference.Predictor(
            weightsDir='./DLToolWeights/HyperIQA',device=device)
        self.transform = self.imageQualityPredictor.transform

    def update(self, imageInfo, topDir):
        imageInfo.update(self.getUpdateDict(imageInfo, topDir))
        return imageInfo
    
    def getUpdateDict(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict(img)
        score_dict.update({'W': width, 'H': height})
        return score_dict

    def update_batch(self, imgs):
        return self.imageQualityPredictor.predict_batch(imgs)
    @staticmethod
    def supportBatchInference():
        return True
    @staticmethod
    def fieldSet():
        return set(['Q512', 'H', 'W'])


class ImageSPAQTool:
    def __init__(self, topDir) -> None:
        self.imageQualityPredictor = SPAQInference.inference_SPAQ.Predictor(
            weightsDir='./DLToolWeights/SPAQ')

    def update(self, imageInfo, topDir):
        imageInfo.update(self.getUpdateDict(imageInfo, topDir))
        return imageInfo
    
    def getUpdateDict(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict(img)
        score_dict.update({'W': width, 'H': height})
        return score_dict
    
    @staticmethod
    def fieldSet():
        return set(['SPAQ', 'H', 'W'])


class WatermarkDetectTool:
    def __init__(self, topDir, device='cuda') -> None:
        self.watermarkPredictor = WatermarkDetectionInference.inference_simple.Predictor(
            weightsDir="./DLToolWeights/WatermarkDetection", device=device)
        self.transform = self.watermarkPredictor.transform

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

    def update_batch(self, imgs):
        watermarkResults = self.watermarkPredictor.predict_batch(imgs)
        return watermarkResults

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def fieldSet():
        return set(['HAS_WATERMARK'])


class SmartCropTool:
    def __init__(self, topDir, device='cuda') -> None:
        self.smartCropPredictor =SmartCropInference.inference.Predictor(
            weightsDir="./DLToolWeights/SmartCrop", device=device)

    def update(self, imageInfo, topDir):
        imageInfo.update(self.getUpdateDict(imageInfo, topDir))
        return imageInfo
    
    def getUpdateDict(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        return self.smartCropPredictor.predict(img)

    @staticmethod
    def supportBatchInference():
        return False

    @staticmethod
    def fieldSet():
        return set(['A_CENTER'])


class ImageOCRTool:
    def __init__(self, topDir, device='cuda',debugOutput=False) -> None:
        self.imageOCRPredictor = OCRInference.inference.Predictor(
            weightsDir="./DLToolWeights/EasyOCR",device=device)
        self.debugOutput = debugOutput

    def update(self, imageInfo, topDir):
        img = OCRInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        if width*height>1024*1024:
            resize_ratio = math.sqrt(1024*1024/(img.size[0]*img.size[1]))
            resizedImg = img.resize(
                tuple(math.ceil(x * resize_ratio) for x in img.size),
                Image.BICUBIC
            )
        else:
            resizedImg = img

        width, height = img.size
        bounds = self.imageOCRPredictor.predict(resizedImg)

        if self.debugOutput:
            bakDir = os.path.join(topDir, 'ocr_result',
                                os.path.dirname(imageInfo['IMG']))
            bakImagePath = os.path.join(
                bakDir, os.path.basename(imageInfo['IMG']))
            if not os.path.exists(bakDir):
                os.makedirs(bakDir)
            if len(bounds) > 0:
                self.draw_boxes(resizedImg, bounds)
                resizedImg.save(bakImagePath)

        imageInfo.update({'W': width, 'H': height, 'TXT':bounds})
        return imageInfo

    def draw_boxes(self, image, bounds, color='yellow', width=6):
        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
        return image
    
    @staticmethod
    def supportBatchInference():
        return False
    
    @staticmethod
    def fieldSet():
        return set(['TXT','H', 'W'])


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
    def __init__(self, topDir, device='cuda') -> None:
        self.imageAestheticPredictor = Aesthetic.Predictor(
            weightsDir='./DLToolWeights/Aesthetic', device=device)
        self.transform = self.imageAestheticPredictor.transform

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageAestheticPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    def update_batch(self, imgs):
        score_dict_list = self.imageAestheticPredictor.predict_batch(imgs)
        return score_dict_list

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def fieldSet():
        return set(['A', 'H', 'W'])


class ImageEATAestheticTool:
    def __init__(self, topDir, device='cuda') -> None:
        self.imageAestheticPredictor = EATInference.inference.Predictor(
            weightsDir='./DLToolWeights/EAT', device=device)
        self.transform = self.imageAestheticPredictor.transform

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        width, height = img.size
        score_dict = self.imageAestheticPredictor.predict(img)
        imageInfo.update({'W': width, 'H': height})
        imageInfo.update(score_dict)
        return imageInfo

    def update_batch(self, imgs):
        score_dict_list = self.imageAestheticPredictor.predict_batch(imgs)
        return score_dict_list

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def fieldSet():
        return set(['A_EAT', 'H', 'W'])


class ImageSRTool:
    def __init__(self, topDir, device='cuda',srType='Photo') -> None:
        if srType=='Photo':
            self.imageSRPredictor = RealESRGANInference.inference_realesrgan.Predictor(
                weightsDir='./DLToolWeights/RealESRGAN',device=device)
        elif srType == 'Anime':
            self.imageSRPredictor = RealCUGANInference.inference_cugan.Predictor(
                weightsDir='./DLToolWeights',device=device)          

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
        return imageArea < 896*896 and imageArea > 384*384 and imageInfo['Q512'] > 60


class ImagePoseEstimateTool:
    def __init__(self, topDir, device='cuda') -> None:
        self.imagePoseEstPredictor = RTMPoseInference.inference.Predictor(
            weightsDir='./DLToolWeights',device=device)

    def update(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))

        preds = self.imagePoseEstPredictor.predict(img)
        # width, height = poseResult.size
        # if width*height >1024*1024:
        #     resize_ratio = math.sqrt(1024*1024/(img.size[0]*img.size[1]))
        #     poseResult = poseResult.resize(
        #                 tuple(math.ceil(x * resize_ratio) for x in img.size),
        #                 Image.BICUBIC
        #             )
        # bakDir = os.path.join(topDir, 'pose_est_result',
        #                       os.path.dirname(imageInfo['IMG']))
        # bakImagePath = os.path.join(
        #     bakDir, os.path.basename(imageInfo['IMG']))
        # if not os.path.exists(bakDir):
        #     os.makedirs(bakDir)
        # poseResult.save(bakImagePath)
        imageInfo.update(preds)
        return imageInfo
    
    def getUpdateDict(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        preds = self.imagePoseEstPredictor.predict(img)
        return preds
    
    @staticmethod
    def fieldSet():
        return set(['POSE_KPTS'])


class ImageObjectDetectTool:
    def __init__(self, topDir, device='cuda',name=None) -> None:
        self.imageObjectDetectPredictor = YoloInference.inference.Predictor(
            weightsDir='./DLToolWeights',weightName=name,device=device)

    def update(self, imageInfo, topDir):
        imageInfo.update(self.getUpdateDict(imageInfo, topDir))
        return imageInfo
    
    def getUpdateDict(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        preds = self.imageObjectDetectPredictor.predict(img)
        return preds
    
    @staticmethod
    def fieldSet():
        return set(['OBJS'])

# class ImageFilterTool:
#     def __init__(self, topDir) -> None:
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k',
#                                                                                cache_dir='./DLToolWeights/OpenCLIP')
#         self.tokenizer = open_clip.get_tokenizer(
#             'ViT-H-14')
#         self.model.to('cuda')

#     def update(self, imageInfo, topDir):
#         img = hpyerIQAInference.inference.pil_loader(
#             os.path.join(topDir, imageInfo['IMG']))

#         img = self.preprocess(img).unsqueeze(0).to('cuda')
#         text = self.tokenizer(
#             ["draft sketch", "finshed work"]).to('cuda')

#         import torch
#         with torch.no_grad(), torch.cuda.amp.autocast():
#             image_features = self.model.encode_image(img)
#             text_features = self.model.encode_text(text)
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
#             text_probs = (100.0 * image_features @
#                           text_features.T).softmax(dim=-1)[0]
#             idx = torch.argmax(text_probs)
#             if idx == 0:
#                 bakDir = os.path.join(topDir, 'raw_before_filter',
#                                       os.path.dirname(imageInfo['IMG']))
#                 rawImagePath = os.path.join(topDir, imageInfo['IMG'])
#                 print('Filter out:%s' % rawImagePath)
#                 bakImagePath = os.path.join(
#                     bakDir, os.path.basename(imageInfo['IMG']))
#                 if not os.path.exists(bakDir):
#                     os.makedirs(bakDir)
#                 move(rawImagePath, bakImagePath)
#             return imageInfo

#     @staticmethod
#     def fieldSet():
#         return set([])


class DeepDanbooruTagTool:
    def __init__(self, topDir, device='cuda') -> None:
        self.imageCaptionPredictor = WDVitTaggerV3.inference.Predictor(
            weightsDir='./DLToolWeights', device=device)
        self.transform = self.imageCaptionPredictor.transform

    def update_batch(self, imgs):
        captionDictListList = self.imageCaptionPredictor.predict_batch(imgs)
        return captionDictListList

    @staticmethod
    def supportBatchInference():
        return True
    
    @staticmethod
    def updateCriteria(imageInfo):
        imageArea = imageInfo['W']*imageInfo['H']
        return imageArea > 384*384 and imageInfo['Q512'] > 35 and \
            ('DBRU_TAG' not in imageInfo.keys() or 'DBRU_TAG' in imageInfo.keys()
             and len(imageInfo['DBRU_TAG']) == 0)

    @staticmethod
    def fieldSet():
        return set(['DBRU_TAG'])
    
class ImageHQCaptionTool:
    def __init__(self, topDir, captionModel='LLAVA', device='cuda') -> None:
        if captionModel == 'MiniCPMLlama3V25':
            self.imageCaptionPredictor = MiniCPMLlama3V25Inference.inference.Predictor(
                weightsDir='./DLToolWeights', device=device)

    def update(self, imageInfo, topDir):
        imageInfo.update(self.getUpdateDict(imageInfo, topDir))
        return imageInfo

    def getUpdateDict(self, imageInfo, topDir):
        img = hpyerIQAInference.inference.pil_loader(
            os.path.join(topDir, imageInfo['IMG']))
        captionDictList = self.imageCaptionPredictor.predict(img)
        return {'HQ_CAP': [captionDict['caption']
                                  for captionDict in captionDictList]}

    def update_batch(self, imgs):
        raise NotImplementedError

    @staticmethod
    def updateCriteria(imageInfo):
        imageArea = imageInfo['W']*imageInfo['H']
        return imageArea > 384*384 and imageInfo['Q512'] > 35 and \
            ('HQ_CAP' not in imageInfo.keys() or 'HQ_CAP' in imageInfo.keys()
             and len(imageInfo['HQ_CAP']) == 0)

    @staticmethod
    def supportBatchInference():
        return False

    @staticmethod
    def fieldSet():
        return set(['HQ_CAP'])

class ImageCaptionTool:
    def __init__(self, topDir, captionModel='LLAVA', device='cuda') -> None:
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
                customCaptionPool=customCaptionPool, weightsDir='./DLToolWeights/BLIP', device=device)
            self.transform = self.imageCaptionPredictor.transform
        elif captionModel == 'BLIP2':
            self.imageCaptionPredictor = BLIP2Inference.inference.Predictor(
                weightsDir='./DLToolWeights', device=device)

    def update(self, imageInfo, topDir):
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

    def update_batch(self, imgs):
        captionDictListList = self.imageCaptionPredictor.predict_batch(imgs)
        return captionDictListList

    @staticmethod
    def updateCriteria(imageInfo):
        imageArea = imageInfo['W']*imageInfo['H']
        return imageArea > 384*384 and imageInfo['Q512'] > 35 and \
            ('CAP' not in imageInfo.keys() or 'CAP' in imageInfo.keys()
             and len(imageInfo['CAP']) == 0)

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def fieldSet():
        return set(['CAP'])


class ImageInfoManager:
    def __init__(self, topDir, imageInfoFileName='ImageInfo.json', 
                 processTools=[], toolConfigYAML=None,topTopDir=None,debugWithoutSave=False,
                 saveInterval=3600) -> None:
        self.topDir = topDir
        self.topTopDir = topTopDir
        self.debugWithoutSave =  debugWithoutSave
        self.processTools = processTools
        self.toolConfigYAML = toolConfigYAML
        self.imageInfoFilePath = os.path.join(self.topDir, imageInfoFileName)
        self.supportImageFormatList = ['.jpg', '.webp', '.png', '.heic']
        self.saveInterval = saveInterval
        if os.path.isfile(self.imageInfoFilePath):
            with open(self.imageInfoFilePath, 'r', encoding='utf8') as f:
                self.imageInfoList = json.load(f)
        else:
            print('ImageInfo File Not Found. Create one.')
            self.imageInfoList = []
        self.createProcessTools()

    def isInFilterDir(self,dir,filteredDirList):
        if self.topTopDir:
            topDir = self.topTopDir
        else:
            topDir = self.topDir
        filteredDirList = [pathlib.Path(dirPath)
                           for dirPath in filteredDirList]
        dirRelativepath = pathlib.Path(
            os.path.relpath(dir, topDir))

        detectedFilterDir = False
        for filterd in filteredDirList:
            if filterd in dirRelativepath.parents:
                detectedFilterDir = True
                break
        return detectedFilterDir
    
    def createProcessTools(self):
        if self.toolConfigYAML:
            print('Use tool config YAML,param processTools has been ignored.')
            with open(self.toolConfigYAML, 'r') as f:
                toolsConfig = yaml.safe_load(f)['Tools']
                toolsConfig = [] if toolsConfig is None else toolsConfig

            processTools = []
            for toolDict in toolsConfig:
                toolDictUpdate = {'forceUpdate': False,'multiGPUs':None,'excludeDirs':None,'includeDirs':None,
                                  'args': {}, 'batchsize': 1, 'num_workers': 4}
                toolDictUpdate.update(toolDict)
                toolDictUpdate['toolClass'] = getattr(
                    sys.modules[__name__], toolDict['toolClass'])
                processTools.append(toolDictUpdate)
            self.processTools = processTools

    def saveImageInfoList(self):
        if self.debugWithoutSave:
            print('!!!DEBUG MODE. NOT SAVED!!!')
            return
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        tempFilePath = self.imageInfoFilePath+'.lock'
        with open(tempFilePath, 'w',encoding='utf8') as f:
            json.dump(self.imageInfoList, f,cls=NpEncoder)
        shutil.move(tempFilePath,self.imageInfoFilePath)
        

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
    @staticmethod
    def processFunc(param):
        itemIdcs,toolClass,topDir,device,imageInfoList,toolArgs = param
        toolArgs['device'] = device
        toolInstance = toolClass(topDir, **toolArgs)
        updateDictIdxList = []
        for i, imageInfoIdx in enumerate(itemIdcs):
            try:
                updateDictIdxList.append((imageInfoIdx,toolInstance.getUpdateDict(
                    imageInfoList[imageInfoIdx], topDir)))
            except Exception as e:
                raise e
                print('ERROR:%s:%s' %
                    (imageInfoList[imageInfoIdx], str(e)))
        return updateDictIdxList
                
    def infoUpdate(self):
        processToolNameListDict = {}
        for processTool in self.processTools:
            processToolClass = processTool['toolClass']
            processToolNameListDict[processToolClass] = {
                'fieldSet': processToolClass.fieldSet(),
                'forceUpdate': processTool['forceUpdate'],
                'args': processTool['args'],
                'batchsize': processTool['batchsize'],
                'num_workers': processTool['num_workers'],
                'multiGPUs': processTool['multiGPUs'],
                'excludeDirs':processTool['excludeDirs'],
                'includeDirs':processTool['includeDirs'],
                'itemIdx': []}

        for processTool, processDict in processToolNameListDict.items():
            toolUpdateCount = 0
            if (processDict['includeDirs'] and \
                    not self.isInFilterDir(self.topDir,processDict['includeDirs'])) \
                or \
                (processDict['excludeDirs'] and self.isInFilterDir(self.topDir,processDict['excludeDirs'])):
                print('Skip %s' % processTool.__name__)
                continue

            for idx, imageInfo in enumerate(self.imageInfoList):
                meetUpdateCriteria = True
                if hasattr(processTool, 'updateCriteria'):
                    meetUpdateCriteria = processDict['forceUpdate'] or processTool.updateCriteria(
                        imageInfo)
                else:
                    meetUpdateCriteria = processDict['forceUpdate'] or len(
                        processDict['fieldSet']-set(imageInfo.keys())) > 0
                if meetUpdateCriteria:
                    processDict['itemIdx'].append(idx)

            if len(processDict['itemIdx']) > 0:
                print('Tool: %s' % processTool.__name__)
                if not processDict['multiGPUs']:
                    toolInstance = processTool(self.topDir, **processDict['args'])
                    if hasattr(processTool, 'supportBatchInference') and processTool.supportBatchInference():
                        ds = BatchInferenceDataset(
                            self.topDir, self.imageInfoList, processDict['itemIdx'], toolInstance.transform)
                        dtldr = DataLoader(ds,
                                        batch_size=processDict['batchsize'],
                                        shuffle=False,
                                        num_workers=processDict['num_workers'],
                                        drop_last=False)
                        with tqdm(total=len(ds)) as pbar:
                            lastTs = time.time()
                            for indices, imgs in dtldr:
                                updateDictList = toolInstance.update_batch(imgs)
                                for imageInfoIdx, updateDict in zip(indices, updateDictList):
                                    self.imageInfoList[imageInfoIdx].update(
                                        updateDict)
                                pbar.update(len(indices))
                                toolUpdateCount = toolUpdateCount+len(indices)
                                nowTs = time.time()
                                if nowTs-lastTs >= self.saveInterval:
                                    lastTs = nowTs
                                    self.saveImageInfoList()
                    else:
                        lastTs = time.time()
                        for i, imageInfoIdx in enumerate(tqdm(processDict['itemIdx'])):
                            try:
                                updateResult = toolInstance.update(
                                    self.imageInfoList[imageInfoIdx], self.topDir)
                                if updateResult is not None:
                                    toolUpdateCount = toolUpdateCount+1
                            except Exception as e:
                                raise e
                                print('ERROR:%s:%s' %
                                    (self.imageInfoList[imageInfoIdx], str(e)))
                            nowTs = time.time()
                            if nowTs-lastTs >= self.saveInterval:
                                lastTs = nowTs
                                if toolUpdateCount>0:
                                    self.saveImageInfoList()
                else:
                    lastTs = time.time()
                    tasks = processDict['itemIdx']
                    woker = processDict['multiGPUs']
                    wokerNum = len(woker)
                    taskNum = len(tasks)
                    divideStep = math.ceil(taskNum/wokerNum)
                    subLists=[(tasks[i:i+divideStep],
                               processTool,
                               self.topDir,
                               'cuda:%d'%devNum,
                               self.imageInfoList,
                               processDict['args']) 
                              for i,devNum  in zip(range(0, taskNum, divideStep),woker)]

                    print("MultiGPUs: %d task(s) assigned to %d workers."%(taskNum,wokerNum))
                    for param in subLists:
                        tasks,_,_,device,_,_ = param
                        print('Worker %s: %d'%(device,len(tasks)))

                    with Pool(wokerNum) as p:
                        for updateDictIdxList in tqdm(p.imap(self.processFunc, subLists), total=len(subLists)):
                            for imageInfoIdx, updateDict in updateDictIdxList:
                                self.imageInfoList[imageInfoIdx].update(updateDict)
                            toolUpdateCount = toolUpdateCount+len(updateDictIdxList)
                            nowTs = time.time()
                            if nowTs-lastTs >= self.saveInterval:
                                lastTs = nowTs
                                self.saveImageInfoList() 
                if toolUpdateCount==0:
                    print(f'No update in {self.topDir} by tool {processTool.__name__}.')
                else:         
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
