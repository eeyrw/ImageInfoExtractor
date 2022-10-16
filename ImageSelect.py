import json
import argparse
from shutil import copyfile
import os
from tqdm import tqdm


class ImageDsCreator:
    def __init__(self, outputDir) -> None:
        self.imageSetList = []
        self.outputDir = outputDir
        self.imageInfoList = []
        self.imageInfoFilePath = os.path.join(self.outputDir,'ImageInfo.json')

    def generate(self):
        totalImageNum = 0
        for jsonPath, criteria in self.imageSetList:
            with open(jsonPath, 'r') as f:
                imageInfo = json.load(f)
            dsRoot = os.path.dirname(jsonPath)
            filterList = [
                singleImageInfo for singleImageInfo in imageInfo if criteria(singleImageInfo)]
            totalImageNum = totalImageNum + len(filterList)
            print('Copy %d images to %s from %s' %
                  (len(filterList), self.outputDir, dsRoot))
            for singleImageInfo in tqdm(filterList):
                orinPath = os.path.join(dsRoot, singleImageInfo['IMG'])
                targetDir = os.path.join(
                    self.outputDir, os.path.dirname(singleImageInfo['IMG']))
                targetPath = os.path.join(
                    self.outputDir, singleImageInfo['IMG'])
                if not os.path.isdir(targetDir):
                    os.makedirs(targetDir)
                copyfile(orinPath, targetPath)
                self.imageInfoList.append(singleImageInfo)
        print('Total image num: %s'%totalImageNum)
        with open(self.imageInfoFilePath, 'w') as f:
            json.dump(self.imageInfoList, f)

    def addImageSet(self, jsonPath, criteria):
        self.imageSetList.append((jsonPath, criteria))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonPath', type=str,
                        default=r'F:\NSFW_DS\MHPHOTOS\ImageInfo.json')
    parser.add_argument('--dsRoot', type=str, default=r'F:\NSFW_DS\MHPHOTOS')
    parser.add_argument('--output', type=str,
                        default=r'F:\NSFW_DS\high_quality')
    parser.add_argument('--threshold', type=float, default=70)
    config = parser.parse_args()

    imgDsCreator = ImageDsCreator(r'F:\NSFW_DS\FinalDs')
    imgDsCreator.addImageSet(r'F:\NSFW_DS\MHPHOTOS\ImageInfo.json',
                             lambda singleImageInfo:
                             singleImageInfo['Q512'] > 65 and singleImageInfo['H']*singleImageInfo['W'] > 700*700)
    imgDsCreator.addImageSet(r"F:\NSFW_DS\twitter_nsfw_data\ImageInfo.json",
                             lambda singleImageInfo:
                             singleImageInfo['Q512'] > 65 and singleImageInfo['H']*singleImageInfo['W'] > 900*900)
    imgDsCreator.addImageSet(r"F:\NSFW_DS\various_source_nsfw_data\ImageInfo.json",
                             lambda singleImageInfo:
                             singleImageInfo['Q512'] > 65 and singleImageInfo['H']*singleImageInfo['W'] > 900*900)
    imgDsCreator.generate()
