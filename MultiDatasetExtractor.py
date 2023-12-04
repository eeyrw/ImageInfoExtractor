import sys
from pillow_heif import register_heif_opener
from tqdm import tqdm
import pathlib
import argparse
import json
import os
from shutil import copyfile, move
from PIL import ImageDraw
from ImageInfoExtractor import ImageInfoManager
from ImageSelect import ImageDsCreator


class MultiDatasetExtractor:
    def __init__(self, topDir) -> None:
        self.topDir = topDir
        self.dirsHasImageInfoJson = []
        self.dirsHasNotImageInfoJson = []

    def scanDir(self, printScanResult=False):
        self.dirsHasImageInfoJson, self.dirsHasNotImageInfoJson = self.detectImageInfoFolder(
            self.topDir)

        if printScanResult:
            print(
                'Scan Result (--: Has ImageInfoJson file, ??: Has not ImageInfoJson file)')
            for dir in self.dirsHasImageInfoJson:
                print('--', dir)
            for dir in self.dirsHasNotImageInfoJson:
                print('??', dir)

    def detectImageInfoFolder(self, path, ImageInfoJsonFileName='ImageInfo.json'):
        dirsHasImageInfoJson = []
        dirsHasNotImageInfoJson = []
        dir_HasImageInfo = {}

        for entry in os.scandir(path):
            file_path = entry.path
            if entry.is_file(follow_symlinks=False) and os.path.basename(file_path) == ImageInfoJsonFileName:
                dirsHasImageInfoJson.append(file_path)
                return dirsHasImageInfoJson, dirsHasNotImageInfoJson
            elif entry.is_dir(follow_symlinks=False):
                dirsHasImageInfoJson_, dirsHasNotImageInfoJson_ = self.detectImageInfoFolder(
                    file_path)
                dirsHasImageInfoJson.extend(dirsHasImageInfoJson_)
                dirsHasNotImageInfoJson.extend(dirsHasNotImageInfoJson_)
                if len(dirsHasImageInfoJson_) > 0:
                    dir_HasImageInfo[file_path] = True
                else:
                    dir_HasImageInfo[file_path] = False

        if len(dirsHasImageInfoJson) > 0:
            for path, hasImageInfo in dir_HasImageInfo.items():
                if not hasImageInfo:
                    dirsHasNotImageInfoJson.append(path)

        return dirsHasImageInfoJson, dirsHasNotImageInfoJson

    def runExtractor(self):
        for path in self.dirsHasImageInfoJson:
            print('====Processing %s' % path)
            dsDir = os.path.dirname(path)
            toolConfig = 'MyExtractionBatchPipeline.yaml'
            imageInfoManager = ImageInfoManager(
                dsDir, toolConfigYAML=toolConfig)
            imageInfoManager.updateImages(
                filteredDirList=['raw_before_sr', 'ocr_result'])
            imageInfoManager.infoUpdate()
            imageInfoManager.saveImageInfoList()

    def genSelectedImageInfoJson(self):
        imgDsCreator = ImageDsCreator(self.topDir)
        def criteria(singleImageInfo): return singleImageInfo['Q512'] > 40 and singleImageInfo['H'] * \
            singleImageInfo['W'] >= 1024 * \
            1024 and singleImageInfo['HAS_WATERMARK'] < 0.7
        for path in self.dirsHasImageInfoJson:
            imgDsCreator.addImageSet(path, criteria, 5000000)

        imgDsCreator.filterImageInfoList()
        imgDsCreator.exportImageInfoList(useJsonl=True)




if __name__ == '__main__':
    multiDsExtractor = MultiDatasetExtractor(r'Dataset top dir')
    multiDsExtractor.scanDir(printScanResult=True)
    multiDsExtractor.runExtractor()
