import pathlib
import zipfile
import os
from ImageInfoExtractor import ImageInfoManager
from ImageSelect import ImageDsCreator

class MultiDatasetExtractor:
    def __init__(self, topDir,debugWithoutSave=False) -> None:
        self.topDir = topDir
        self.debugWithoutSave = debugWithoutSave
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

    def runExtractor(self,toolConfig):
        for path in self.dirsHasImageInfoJson:
            print('====Processing %s' % path)
            dsDir = os.path.dirname(path)
            imageInfoManager = ImageInfoManager(
                dsDir, toolConfigYAML=toolConfig,topTopDir=self.topDir,debugWithoutSave=self.debugWithoutSave)
            imageInfoManager.updateImages(
                filteredDirList=['raw_before_sr', 'ocr_result'])
            imageInfoManager.infoUpdate()
            #imageInfoManager.saveImageInfoList()

        for path in self.dirsHasNotImageInfoJson:
            print('====Processing %s' % path)
            dsDir = path
            imageInfoManager = ImageInfoManager(
                dsDir, toolConfigYAML=toolConfig,topTopDir=self.topDir,debugWithoutSave=self.debugWithoutSave)
            imageInfoManager.updateImages(
                filteredDirList=['raw_before_sr', 'ocr_result'])
            imageInfoManager.infoUpdate()
            #imageInfoManager.saveImageInfoList()

    def isInFilterDir(self,dir,filteredDirList):

        filteredDirList = [pathlib.Path(dirPath)
                           for dirPath in filteredDirList]
        dirRelativepath = pathlib.Path(
            os.path.relpath(dir, self.topDir))

        detectedFilterDir = False
        for filterd in filteredDirList:
            if filterd in dirRelativepath.parents:
                detectedFilterDir = True
                break
        return detectedFilterDir
            

    def genSelectedImageInfoJson(self,outputName='ImageInfoSelected.json',filterDirList=[]):
        imgDsCreator = ImageDsCreator(self.topDir)
        def criteria(singleImageInfo): return singleImageInfo['Q512'] > 50 and \
            singleImageInfo['A_EAT'] > 5.5 and singleImageInfo['H'] * \
            singleImageInfo['W'] >= 896 * \
            896
        for path in self.dirsHasImageInfoJson:
            if not self.isInFilterDir(os.path.dirname(path),filterDirList):
                imgDsCreator.addImageSet(path, criteria, 5000000)
            else:
                print(f'Skip {path}')

        imgDsCreator.filterImageInfoList()
        imgDsCreator.exportImageInfoList(jsonName=outputName,useJsonl=False)

    def packImageInfoJsonFiles(self, outputTarPath):
        with zipfile.ZipFile(outputTarPath+'.zip','w') as myzip:
            for path in self.dirsHasImageInfoJson:
                myzip.write(path,compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    multiDsExtractor = MultiDatasetExtractor(r'Dataset top dir')
    multiDsExtractor.scanDir(printScanResult=True)
    multiDsExtractor.runExtractor()
