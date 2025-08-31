from pathlib import Path
import zipfile
import os
from ImageInfoExtractor import ImageInfoManager
from ImageSelect import ImageDsCreator


class MultiDatasetExtractor:
    def __init__(self, topDir, debugWithoutSave=False) -> None:
        self.topDir = topDir
        self.debugWithoutSave = debugWithoutSave
        self.imageInfoFilesInDir = []
        self.dirsHasNotImageInfo = []

    def scanDir(self, printScanResult=False):
        self.imageInfoFilesInDir, self.dirsHasNotImageInfo = self.detectImageInfoFolder(
            self.topDir)

        if printScanResult:
            print(
                'Scan Result (--: Has ImageInfofile, ??: Has not ImageInfofile)')
            for dir in self.imageInfoFilesInDir:
                print('--', dir)
            for dir in self.dirsHasNotImageInfo:
                print('??', dir)

    def detectImageInfoFolder(self, path, ImageInfoFileNameList=['ImageInfo.json', 'ImageInfo.parquet']):
        imageInfoFilesInDir = []
        dirsHasNotImageInfo = []
        dir_HasImageInfo = {}

        current_path = Path(path)
        for imageInfoFile in ImageInfoFileNameList:
            imageInfoPath = current_path/Path(imageInfoFile)
            if imageInfoPath.is_file():
                imageInfoFilesInDir.append(imageInfoPath)

        if len(imageInfoFilesInDir) > 0:
            return (imageInfoFilesInDir,), dirsHasNotImageInfo
        else:
            for entry in current_path.iterdir():
                if entry.is_dir():
                    imageInfoFilesInDir_, dirsHasNotImageInfo_ = self.detectImageInfoFolder(
                        entry)
                    imageInfoFilesInDir.extend(imageInfoFilesInDir_)
                    dirsHasNotImageInfo.extend(dirsHasNotImageInfo_)
                    if len(imageInfoFilesInDir_) > 0:
                        dir_HasImageInfo[entry] = True
                    else:
                        dir_HasImageInfo[entry] = False

            if len(imageInfoFilesInDir) > 0:
                for path, hasImageInfo in dir_HasImageInfo.items():
                    if not hasImageInfo:
                        dirsHasNotImageInfo.append(path)

            return imageInfoFilesInDir, dirsHasNotImageInfo

    def runExtractor(self, toolConfig):
        for imageInfoFiles in self.imageInfoFilesInDir:

            for imageInfoFile_ in imageInfoFiles:
                imageInfoFile = imageInfoFile_
                if imageInfoFile.suffix == '.parquet':
                    break
            print('====Processing %s' % imageInfoFile)

            imageInfoManager = ImageInfoManager(
                imageInfoFile.parent, toolConfigYAML=toolConfig,
                topTopDir=self.topDir, debugWithoutSave=self.debugWithoutSave)
            imageInfoManager.updateImages(
                filteredDirList=['raw_before_sr', 'ocr_result'])
            imageInfoManager.infoUpdate()

        for path in self.dirsHasNotImageInfo:
            print('====Processing %s' % path)
            dsDir = path
            imageInfoManager = ImageInfoManager(
                dsDir, toolConfigYAML=toolConfig, topTopDir=self.topDir, debugWithoutSave=self.debugWithoutSave)
            imageInfoManager.updateImages(
                filteredDirList=['raw_before_sr', 'ocr_result'])
            imageInfoManager.infoUpdate()

    def isInFilterDir(self, dir, filteredDirList):

        filteredDirList = [Path(dirPath)
                           for dirPath in filteredDirList]
        dirRelativepath = Path(
            os.path.relpath(dir, self.topDir))

        detectedFilterDir = False
        for filterd in filteredDirList:
            if filterd in dirRelativepath.parents:
                detectedFilterDir = True
                break
        return detectedFilterDir

    def genSelectedImageInfo(self, outputName='ImageInfoSelected.json', filterDirList=[]):
        imgDsCreator = ImageDsCreator(self.topDir)

        def criteria(singleImageInfo): return singleImageInfo['Q512'] > 50 and \
            singleImageInfo['A_EAT'] > 5.5 and singleImageInfo['H'] * \
            singleImageInfo['W'] >= 896 * \
            896
        for path in self.imageInfoFilesInDir:
            if not self.isInFilterDir(os.path.dirname(path), filterDirList):
                imgDsCreator.addImageSet(path, criteria, 5000000)
            else:
                print(f'Skip {path}')

        imgDsCreator.filterImageInfoList()
        imgDsCreator.exportImageInfoList(jsonName=outputName, useJsonl=False)

    def packImageInfoFiles(self, outputTarPath):
        with zipfile.ZipFile(outputTarPath+'.zip', 'w') as myzip:
            for path in self.imageInfoFilesInDir:
                myzip.write(path, compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    multiDsExtractor = MultiDatasetExtractor(r'Dataset top dir')
    multiDsExtractor.scanDir(printScanResult=True)
    multiDsExtractor.runExtractor()
