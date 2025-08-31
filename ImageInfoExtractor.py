import itertools
import sys
from typing import Iterable
from pillow_heif import register_heif_opener
from tqdm import tqdm
import pathlib
import argparse
import json
import os
import yaml
from PIL import Image
from shutil import copyfile, move
import math
import OCRInference.inference
import YoloInference.inference
import polars as pl

from PIL import ImageDraw
from pathlib import Path, PurePath
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
import shutil
from torch.multiprocessing import Pool, Process, set_start_method

register_heif_opener()


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


CURRENT_SCHEMA = {'IMG': pl.String, 'W': pl.Int32, 'H': pl.Int32,
                  'Q512': pl.Float32,
                  'CAP': pl.List(pl.String),
                  'A': pl.Float32,
                  'A_EAT': pl.Float32,
                  'HQ_CAP': pl.List(pl.String),
                  'A_CENTER': pl.List(pl.Float32),
                  'DBRU_TAG': pl.String,
                  'POSE_KPTS': pl.List(pl.Struct({'BBOX': pl.List(pl.Float32), 'INVLD_KPTS_IDX': pl.List(pl.Int32), 'KPTS_X': pl.List(pl.Float32), 'KPTS_Y': pl.List(pl.Float32)})),
                  'HAS_WATERMARK': pl.Float32,
                  'IMG_EMBD': pl.List(pl.Float32)}


class BatchInferenceDataset(Dataset):
    def __init__(self, topDir, imageInfoDF, indexList: pl.Series, transform):
        self.transform = transform
        self.imageInfoDF = imageInfoDF
        self.indexList = indexList
        self.topDir = topDir

    def __len__(self):
        return len(self.indexList)

    def __getitem__(self, item):
        idx = self.indexList[item]
        image_path = os.path.join(self.topDir,
                                  self.imageInfoDF['IMG'][idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            x = self.transform(image)
        else:
            x = image
        return idx, x


class ImageSizeInfoCorrectTool:
    def __init__(self, topDir) -> None:
        pass

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        with open(os.path.join(topDir, imageInfoDF['IMG'][idx]), 'rb') as f:
            img = Image.open(f)
        width, height = img.size
        ret = None
        if 'W' in imageInfoDF.columns and 'H' in imageInfoDF.columns:
            if width != imageInfoDF['W'][idx] or height != imageInfoDF['H'][idx]:
                print('Correct size: %s' % imageInfoDF['IMG'][idx])
                ret = {'W': width, 'H': height}
        else:
            print('Create size: %s' % imageInfoDF['IMG'][idx])
            ret = {'W': width, 'H': height}
        return ret

    @staticmethod
    def fieldSet():
        return set(['H', 'W'])


class ImageQuailityTool:
    def __init__(self, topDir, device='cuda') -> None:
        import hpyerIQAInference.inference
        self.imageQualityPredictor = hpyerIQAInference.inference.Predictor(
            weightsDir='./DLToolWeights/HyperIQA', device=device)
        self.transform = self.imageQualityPredictor.transform

    def getUpdateDict(self, imageInfoDF, idx,  topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
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


class WatermarkDetectTool:
    def __init__(self, topDir, device='cuda') -> None:
        import WatermarkDetectionInference.inference_simple
        self.watermarkPredictor = WatermarkDetectionInference.inference_simple.Predictor(
            weightsDir="./DLToolWeights/WatermarkDetection", device=device)
        self.transform = self.watermarkPredictor.transform

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        return self.watermarkPredictor.predict(img)

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
        import SmartCropInference.inference
        self.smartCropPredictor = SmartCropInference.inference.Predictor(
            weightsDir="./DLToolWeights/SmartCrop", device=device)

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        return self.smartCropPredictor.predict(img)

    @staticmethod
    def supportBatchInference():
        return False

    @staticmethod
    def fieldSet():
        return set(['A_CENTER'])


class ImageOCRTool:
    def __init__(self, topDir, device='cuda', debugOutput=False) -> None:
        self.imageOCRPredictor = OCRInference.inference.Predictor(
            weightsDir="./DLToolWeights/EasyOCR", device=device)
        self.debugOutput = debugOutput

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = OCRInference.inference.pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        width, height = img.size
        if width*height > 1024*1024:
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
                                  os.path.dirname(imageInfoDF['IMG'][idx]))
            bakImagePath = os.path.join(
                bakDir, os.path.basename(imageInfoDF['IMG'][idx]))
            if not os.path.exists(bakDir):
                os.makedirs(bakDir)
            if len(bounds) > 0:
                self.draw_boxes(resizedImg, bounds)
                resizedImg.save(bakImagePath)

        return {'W': width, 'H': height, 'TXT': bounds}

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
        return set(['TXT', 'H', 'W'])


class JpegQuailityTool:
    def __init__(self, topDir) -> None:
        import FBCNNInference.inference
        self.imageQualityPredictor = FBCNNInference.inference.Predictor(
            weightsDir='./DLToolWeights/FBCNN')

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        width, height = img.size
        score_dict = self.imageQualityPredictor.predict(img)
        score_dict.update({'W': width, 'H': height})
        return score_dict

    @staticmethod
    def fieldSet():
        return set(['QF', 'H', 'W'])


class ImageEATAestheticTool:
    def __init__(self, topDir, device='cuda') -> None:
        import EATInference.inference
        self.imageAestheticPredictor = EATInference.inference.Predictor(
            weightsDir='./DLToolWeights/EAT', device=device)
        self.transform = self.imageAestheticPredictor.transform

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        width, height = img.size
        score_dict = self.imageAestheticPredictor.predict(img)
        score_dict.update({'W': width, 'H': height})
        return score_dict

    def update_batch(self, imgs):
        score_dict_list = self.imageAestheticPredictor.predict_batch(imgs)
        return score_dict_list

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def fieldSet():
        return set(['A_EAT', 'H', 'W'])


class ImageEmbeddingTool:
    def __init__(self, topDir, device='cuda') -> None:
        import DINOv3Inference.inference
        self.imageEmbeddingPredictor = DINOv3Inference.inference.Predictor(
            weightsDir='./DLToolWeights', device=device)
        self.transform = None

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        width, height = img.size
        embed_dict = self.imageEmbeddingPredictor.predict(img)
        embed_dict.update({'W': width, 'H': height})

        return embed_dict

    @staticmethod
    def supportBatchInference():
        return False

    @staticmethod
    def fieldSet():
        return set(['IMG_EMBD', 'H', 'W'])


class ImageSRTool:
    def __init__(self, topDir, device='cuda', srType='Photo') -> None:
        if srType == 'Photo':
            import RealESRGANInference.inference_realesrgan
            self.imageSRPredictor = RealESRGANInference.inference_realesrgan.Predictor(
                weightsDir='./DLToolWeights/RealESRGAN', device=device)
        elif srType == 'Anime':
            import RealCUGANInference.inference_cugan
            self.imageSRPredictor = RealCUGANInference.inference_cugan.Predictor(
                weightsDir='./DLToolWeights', device=device)

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
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
                              os.path.dirname(imageInfoDF['IMG'][idx]))
        rawImagePath = os.path.join(topDir, imageInfoDF['IMG'][idx])
        bakImagePath = os.path.join(
            bakDir, os.path.basename(imageInfoDF['IMG'][idx]))
        if not os.path.exists(bakDir):
            os.makedirs(bakDir)
        copyfile(rawImagePath, bakImagePath)
        savedPath = rawImagePath
        srImg.save(savedPath)
        width, height = srImg.size

        return {'W': width, 'H': height}

    @staticmethod
    def fieldSet():
        return set(['H', 'W'])

    @staticmethod
    def updateFilter(imageInfoDF):
        mask = (
            (pl.col("W") * pl.col("H") < 896*896)
            & (pl.col("W") * pl.col("H") > 384 * 384)
            & (pl.col("Q512") > 60)
        )
        return imageInfoDF.filter(mask).get_column("IDX")


class ImagePoseEstimateTool:
    def __init__(self, topDir, device='cuda') -> None:
        import RTMPoseInference.inference
        self.imagePoseEstPredictor = RTMPoseInference.inference.Predictor(
            weightsDir='./DLToolWeights', device=device)

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        preds = self.imagePoseEstPredictor.predict(img)
        return preds

    @staticmethod
    def fieldSet():
        return set(['POSE_KPTS'])


class ImageObjectDetectTool:
    def __init__(self, topDir, device='cuda', name=None) -> None:
        self.imageObjectDetectPredictor = YoloInference.inference.Predictor(
            weightsDir='./DLToolWeights', weightName=name, device=device)
        self.transform = self.imageObjectDetectPredictor.transform

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        preds = self.imageObjectDetectPredictor.predict(img)
        return preds

    def update_batch(self, imgs):
        return self.imageObjectDetectPredictor.predict_batch(imgs)

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def custom_collate(original_batch):
        trans = list(map(list, itertools.zip_longest(
            *original_batch, fillvalue=None)))
        return trans

    @staticmethod
    def fieldSet():
        return set(['OBJS'])


class DeepDanbooruTagTool:
    def __init__(self, topDir, device='cuda') -> None:
        import WDVitTaggerV3.inference
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
    def updateFilter(imageInfoDF):
        """
        返回符合条件的索引 (pl.Series)，索引列为 'IDX'
        条件:
        - 面积 > 384*384
        - Q512 > 35
        - DBRU_TAG 列不存在 或 为空
        """
        # 如果没有 DBRU_TAG 列，则先加一列默认 None
        if "DBRU_TAG" not in imageInfoDF.columns:
            df = imageInfoDF.with_columns(pl.lit(None).alias("DBRU_TAG"))
        else:
            df = imageInfoDF

        mask = (
            (pl.col("W") * pl.col("H") > 384 * 384)
            & (pl.col("Q512") > 35)
            & (pl.col("DBRU_TAG").is_null())
        )

        # 返回 pl.Series (索引列)
        return df.filter(mask).get_column("IDX")

    @staticmethod
    def fieldSet():
        return set(['DBRU_TAG'])


class ImageHQCaptionTool:
    def __init__(self, topDir, captionModel='LLAVA', device='cuda') -> None:
        if captionModel == 'MiniCPMLlama3V25':
            import MiniCPMLlama3V25Inference.inference
            self.imageCaptionPredictor = MiniCPMLlama3V25Inference.inference.Predictor(
                weightsDir='./DLToolWeights', device=device)

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        captionDictList = self.imageCaptionPredictor.predict(img)
        return {'HQ_CAP': [captionDict['caption']
                           for captionDict in captionDictList]}

    def update_batch(self, imgs):
        raise NotImplementedError

    @staticmethod
    def updateFilter(imageInfoDF):
        if "HQ_CAP" not in imageInfoDF.columns:
            df = imageInfoDF.with_columns(pl.lit(None).alias("HQ_CAP"))
        else:
            df = imageInfoDF

        mask = (
            (pl.col("W") * pl.col("H") > 384 * 384)
            & (pl.col("Q512") > 35)
            & (pl.col("HQ_CAP").is_null())
        )
        return df.filter(mask).get_column("IDX")

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
            import BLIPInference.predict_simple
            self.imageCaptionPredictor = BLIPInference.predict_simple.Predictor(
                customCaptionPool=customCaptionPool, weightsDir='./DLToolWeights/BLIP', device=device)
            self.transform = self.imageCaptionPredictor.transform
        elif captionModel == 'BLIP2':
            import BLIP2Inference.inference
            self.imageCaptionPredictor = BLIP2Inference.inference.Predictor(
                weightsDir='./DLToolWeights', device=device)

    def getUpdateDict(self, imageInfoDF, idx, topDir):
        img = pil_loader(
            os.path.join(topDir, imageInfoDF['IMG'][idx]))
        captionDictList = self.imageCaptionPredictor.predict(img)
        updateDict = {'CAP': [captionDict['caption']
                              for captionDict in captionDictList]}
        havePrintFileName = False
        for captionDict in captionDictList:
            if captionDict['isCustomCap']:
                if not havePrintFileName:
                    print('File:'+imageInfoDF['IMG'][idx])
                    havePrintFileName = True
                print('Custom cap: rank %s cap %s' %
                      (captionDict['rank'], captionDict['caption']))
        return updateDict

    def update_batch(self, imgs):
        captionDictListList = self.imageCaptionPredictor.predict_batch(imgs)
        return captionDictListList

    @staticmethod
    def updateFilter(imageInfoDF):
        if "CAP" not in imageInfoDF.columns:
            df = imageInfoDF.with_columns(pl.lit(None).alias("CAP"))
        else:
            df = imageInfoDF

        mask = (
            (pl.col("W") * pl.col("H") > 384 * 384)
            & (pl.col("Q512") > 35)
            & (pl.col("CAP").is_null())
        )
        return df.filter(mask).get_column("IDX")

    @staticmethod
    def supportBatchInference():
        return True

    @staticmethod
    def fieldSet():
        return set(['CAP'])


import time

class UpdateBuffer:
    def __init__(self, df, batch_update_func, buffer_threshold=1000, save_func=None, save_interval=None):
        """
        批量更新缓冲器

        用途：
            - 避免频繁对 DataFrame 逐条更新，提升性能
            - 支持数量阈值 + 时间阈值 两种策略自动 flush
            - 可以在任务完成时强制 flush，保证结果不会丢失

        参数:
            df : object
                需要更新的 DataFrame 引用（比如 self.imageInfoDF）
            batch_update_func : callable
                用于批量更新 DataFrame 的函数，格式:
                (df, indices, updates) -> df
            buffer_threshold : int, 默认 1000
                缓冲区容量阈值，超过时会触发一次 flush
            save_func : callable, 可选
                每次 flush 后调用的保存函数，例如 self.saveImageInfoList
            save_interval : float, 可选
                时间阈值（秒），超过该时间即使 buffer 未满也会触发 flush
        """
        self.df = df
        self.batch_update = batch_update_func
        self.buffer_threshold = buffer_threshold
        self.save_func = save_func
        self.save_interval = save_interval
        self.buffer = []                   # 存放 (index, update_dict) 的列表
        self.last_flush_time = time.time() # 上一次 flush 的时间戳

    def add(self, indices, updates):
        """
        向缓冲区添加一条或多条更新记录。
        - 如果超过数量阈值，自动 flush
        - 如果超过时间阈值，自动 flush

        参数:
            indices : int | list[int] | tuple[int]
                要更新的 DataFrame 行索引（可以是单个或批量）
            updates : dict | list[dict]
                对应索引的更新内容（单条或批量）
        """
        # 兼容单条和批量情况
        if isinstance(indices, Iterable):
            self.buffer.extend(zip(indices, updates))
        else:
            self.buffer.append((indices, updates))

        # 判断是否需要触发 flush
        now = time.time()
        need_flush = (
            len(self.buffer) >= self.buffer_threshold or   # 数量条件
            (self.save_interval and now - self.last_flush_time >= self.save_interval) # 时间条件
        )
        if need_flush:
            self.flush()

    def flush(self, force=False):
        """
        将缓冲区的更新应用到 DataFrame 并清空缓冲区。

        参数:
            force : bool, 默认 False
                是否强制写回。即使 buffer 不满、时间没到，也会立刻 flush。

        逻辑:
            - 如果 buffer 为空，直接返回
            - 调用 batch_update_func 批量更新 DF
            - 清空 buffer
            - 更新时间戳
            - 如果提供了 save_func，调用保存
        """
        if not self.buffer and not force:
            return

        if self.buffer:
            indices, updates = zip(*self.buffer)
            self.df = self.batch_update(self.df, indices, updates)
            self.buffer.clear()

        self.last_flush_time = time.time()

        # 如果有保存函数，每次 flush 后调用一次
        if self.save_func:
            self.save_func(self)

    def get_df(self):
        """
        返回当前最新的 DataFrame。
        注意：在使用前最好先手动 flush(force=True)，确保缓存写回。
        """
        return self.df
    

class ImageInfoManager:
    def __init__(self, topDir,
                 imageInfoFileName='ImageInfo.json',
                 processTools=[], toolConfigYAML=None, topTopDir=None, debugWithoutSave=False,
                 saveInterval=3600) -> None:
        self.topDir = Path(topDir)
        self.topTopDir = Path(topTopDir)
        self.debugWithoutSave = debugWithoutSave
        self.processTools = processTools
        self.toolConfigYAML = toolConfigYAML
        self.imageInfoFilePath = self.topDir/imageInfoFileName
        self.supportImageFormatList = ['.jpg', '.webp', '.png', '.heic']
        self.saveInterval = saveInterval

        self.createProcessTools()

        if not self.imageInfoFilePath.is_file():
            print('Image info file not found. Creating empty DataFrame.')
            self.imageInfoDF = pl.DataFrame([])
        else:
            ext = self.imageInfoFilePath.suffix.lower()
            if ext == ".json":
                self.imageInfoDF = pl.read_json(
                    self.imageInfoFilePath, schema=CURRENT_SCHEMA)
            elif ext == ".parquet":
                self.imageInfoDF = pl.read_parquet(self.imageInfoFilePath)
            else:
                print(
                    f"Unsupported file type {ext}. Creating empty DataFrame.")
                self.imageInfoDF = pl.DataFrame([])

    def isInFilterDir(self, dir, filteredDirList):
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
                toolDictUpdate = {'forceUpdate': False, 'multiGPUs': None, 'excludeDirs': None, 'includeDirs': None,
                                  'args': {}, 'batchsize': 1, 'num_workers': 4}
                toolDictUpdate.update(toolDict)
                toolDictUpdate['toolClass'] = getattr(
                    sys.modules[__name__], toolDict['toolClass'])
                processTools.append(toolDictUpdate)
            self.processTools = processTools

    def saveImageInfoList(self,updateJson=False):
        if self.debugWithoutSave:
            print('!!!DEBUG MODE. NOT SAVED!!!')
            return

        if 'IDX' in self.imageInfoDF.columns:
            df_to_save = self.imageInfoDF.drop('IDX')  # 删除第一列
        else:
            df_to_save = self.imageInfoDF

        if updateJson:
            tmpFile = self.imageInfoFilePath.with_suffix('.json.lock')
            targetFile = self.imageInfoFilePath.with_suffix('.json')
            with open(tmpFile, 'w', encoding='utf8') as f:
                df_to_save.write_json(f)
            shutil.move(tmpFile, targetFile)


        tmpFile = self.imageInfoFilePath.with_suffix('.parquet.lock')
        targetFile = self.imageInfoFilePath.with_suffix('.parquet')

        with open(tmpFile, 'w') as f:
            df_to_save.write_parquet(f)
        shutil.move(tmpFile, targetFile)

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
        itemIdcs, toolClass, topDir, device, imageInfoList, toolArgs = param
        toolArgs['device'] = device
        toolInstance = toolClass(topDir, **toolArgs)
        updateDictIdxList = []
        for i, imageInfoIdx in enumerate(itemIdcs):
            try:
                updateDictIdxList.append((imageInfoIdx, toolInstance.getUpdateDict(
                    imageInfoList[imageInfoIdx], topDir)))
            except Exception as e:
                raise e
                print('ERROR:%s:%s' %
                      (imageInfoList[imageInfoIdx], str(e)))
        return updateDictIdxList

    def smartPrint(self, items, desc, maxShowNum=10):
        itemNum = len(items)
        if itemNum > 0:
            displayCounter = maxShowNum
            for item in items:
                if displayCounter > 0:
                    displayCounter = displayCounter - 1
                    print(f'{desc}:{item}')
                elif displayCounter == 0:
                    print(f'{desc}: {itemNum} items to be displayed. Too much to show...')
                    break

    def batch_update(self, df: pl.DataFrame, idx_series: pl.Series, updates_list: list[dict]) -> pl.DataFrame:
        """
        向量化批量更新 DataFrame。
        df: 原始 DataFrame
        idx_series: 待更新行的索引 Series（和 updates_list 一一对应）
        updates_list: 更新字典列表，每个字典对应一行更新
        """
        if len(idx_series) != len(updates_list):
            raise ValueError("idx_series 和 updates_list 长度必须一致")

        idx_series = pl.Series(idx_series)

        # 临时 DataFrame 保存更新值
        updates_df = pl.DataFrame(
            updates_list, schema_overrides=CURRENT_SCHEMA)

        # 哪些列会从 updates_df 来
        upd_cols = set(updates_df.columns)

        # 拼接 idx_series 在最前面
        updates_df = pl.DataFrame(
            [idx_series.rename("IDX")]).hstack(updates_df)

        # left join 保留 df 的所有行（updates_df 的 IDX 是 df 的子集）
        dfj = df.join(updates_df, on="IDX", how="left", suffix="_upd")

        # 对 df 里已有的列：
        # - 如果该列在 updates_df 里，用 coalesce([col_upd, col])：有更新值就覆盖，否则保持原值
        # - 否则直接保留原列
        cols = [pl.col("IDX")] + [
            pl.coalesce([pl.col(f"{c}_upd"), pl.col(c)]).alias(
                c) if c in upd_cols else pl.col(c)
            for c in df.columns if c != "IDX"
        ]

        # updates_df 里原来没有的列（新增列）直接追加
        new_cols = [
            pl.col(c) for c in updates_df.columns if c not in df.columns and c != "IDX"]

        return dfj.select([*cols, *new_cols])

    def find_missing_or_null_indices(self, df: pl.DataFrame, fields: list[str], idxField='IDX') -> pl.Series:
        idxs = set()

        for c in fields:
            if c not in df.columns:
                # 字段缺失 -> 全部行
                idxs.update(df[idxField].to_list())
            else:
                # 字段存在 -> 取该列为 null 的行
                null_idx = df.filter(df[c].is_null())[idxField].to_list()
                idxs.update(null_idx)

        # 返回 UInt32 索引 Series
        return pl.Series("idx", sorted(idxs), dtype=pl.UInt32)

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
                'excludeDirs': processTool['excludeDirs'],
                'includeDirs': processTool['includeDirs'],
                'itemIdx': []}

        for processTool, processDict in processToolNameListDict.items():
            toolUpdateCount = 0
            if (processDict['includeDirs'] and
                not self.isInFilterDir(self.topDir, processDict['includeDirs'])) \
                    or \
                    (processDict['excludeDirs'] and self.isInFilterDir(self.topDir, processDict['excludeDirs'])):
                print('Skip %s' % processTool.__name__)
                continue

            if len(self.imageInfoDF)==0:
                print(f'Skip empty dataset dir {self.topDir}')
                break

            if processDict['forceUpdate']:
                filteredImageInfoIdcs = self.imageInfoDF['IDX']
            else:
                if hasattr(processTool, 'updateFilter'):
                    filteredImageInfoIdcs = processTool.updateFilter(
                        self.imageInfoDF)
                else:
                    filteredImageInfoIdcs = self.find_missing_or_null_indices(
                        self.imageInfoDF, processDict['fieldSet'])

            processDict['itemIdx'] = filteredImageInfoIdcs

            if len(processDict['itemIdx']) > 0:
                print('Tool: %s' % processTool.__name__)

                def saveFunc(bufferObj):
                    self.imageInfoDF = bufferObj.get_df()
                    self.saveImageInfoList()

                buffer = UpdateBuffer(
                    df=self.imageInfoDF,
                    batch_update_func=self.batch_update,
                    buffer_threshold=10000,
                    save_func=saveFunc,
                    save_interval=self.saveInterval
                )

                if not processDict['multiGPUs']:
                    toolInstance = processTool(
                        self.topDir, **processDict['args'])
                    if hasattr(processTool, 'supportBatchInference') and processTool.supportBatchInference():
                        ds = BatchInferenceDataset(
                            self.topDir, self.imageInfoDF, processDict['itemIdx'], toolInstance.transform)

                        if hasattr(processTool, 'custom_collate'):
                            custom_collate = processTool.custom_collate
                        else:
                            custom_collate = None
                        dtldr = DataLoader(ds,
                                           batch_size=processDict['batchsize'],
                                           shuffle=False,
                                           num_workers=processDict['num_workers'],
                                           collate_fn=custom_collate,
                                           drop_last=False)
                        with tqdm(total=len(ds)) as pbar:
                            for indices, imgs in dtldr:
                                updateDictList = toolInstance.update_batch(
                                    imgs)
                                buffer.add(indices, updateDictList)
                                pbar.update(len(indices))
                                toolUpdateCount = toolUpdateCount+len(indices)
                    else:
                        for i, imageInfoIdx in enumerate(tqdm(processDict['itemIdx'])):
                            try:
                                updateResult = toolInstance.getUpdateDict(
                                    self.imageInfoDF, imageInfoIdx, self.topDir)
                                buffer.add(i, updateResult)
                                if updateResult is not None:
                                    toolUpdateCount = toolUpdateCount+1
                            except Exception as e:
                                raise e
                                print('ERROR:%s:%s' %
                                      (self.imageInfoList[imageInfoIdx], str(e)))
                else:
                    tasks = processDict['itemIdx']
                    woker = processDict['multiGPUs']
                    wokerNum = len(woker)
                    taskNum = len(tasks)
                    divideStep = math.ceil(taskNum/wokerNum)
                    subLists = [(tasks[i:i+divideStep],
                                 processTool,
                                 self.topDir,
                                 'cuda:%d' % devNum,
                                self.imageInfoList,
                                processDict['args'])
                                for i, devNum in zip(range(0, taskNum, divideStep), woker)]

                    print("MultiGPUs: %d task(s) assigned to %d workers." %
                          (taskNum, wokerNum))
                    for param in subLists:
                        tasks, _, _, device, _, _ = param
                        print('Worker %s: %d' % (device, len(tasks)))

                    with Pool(wokerNum) as p:
                        for updateDictIdxList in tqdm(p.imap(self.processFunc, subLists), total=len(subLists)):
                            for imageInfoIdx, updateDict in updateDictIdxList:
                                buffer.add(imageInfoIdx,updateDict)
                            toolUpdateCount = toolUpdateCount + \
                                len(updateDictIdxList)
                if toolUpdateCount == 0:
                    print(
                        f'No update in {self.topDir} by tool {processTool.__name__}.')
                else:
                    buffer.flush(force=True)
            else:
                print('No update by %s' % processTool.__name__)
                continue

    def updateImages(self, filteredDirList=[]):
        # 1️⃣ 扫描磁盘图片
        actualList = self.getImageList(filteredDirList, relPath=True)
        actualDF = pl.DataFrame({'IMG': actualList})

        if "IDX" in self.imageInfoDF.columns:
            self.imageInfoDF = self.imageInfoDF.drop("IDX")

        deleted_list = []

        if 'IMG' not in self.imageInfoDF.columns or self.imageInfoDF.is_empty():
            self.imageInfoDF = actualDF.with_row_index(name="IDX")
            new_list = actualList
        else:
            # 2️⃣ 找出删除的图片
            deletedDF = self.imageInfoDF.filter(
                ~pl.col('IMG').is_in(actualDF['IMG']))
            deleted_list = deletedDF['IMG'].to_list()

            # 3️⃣ 删除不存在的图片
            self.imageInfoDF = self.imageInfoDF.filter(
                pl.col('IMG').is_in(actualDF['IMG']))

            # 4️⃣ 找出新增图片
            newImagesDF = actualDF.filter(
                ~pl.col('IMG').is_in(self.imageInfoDF['IMG']))
            new_list = newImagesDF['IMG'].to_list()

            # 5️⃣ 为新增图片补齐列
            for col in self.imageInfoDF.columns:
                if col != 'IMG' and col not in newImagesDF.columns:
                    newImagesDF = newImagesDF.with_columns(
                        pl.lit(None).alias(col))

            # 6️⃣ 合并
            self.imageInfoDF = pl.concat(
                [self.imageInfoDF, newImagesDF], how='vertical').rechunk()

            self.imageInfoDF = self.imageInfoDF.with_row_index(name="IDX")

        # 7️⃣ 打印新增和删除图片信息
        self.smartPrint(new_list,'Add')
        self.smartPrint(deleted_list,'Del')


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
