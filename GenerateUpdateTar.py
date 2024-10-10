import pathlib
import os
from tqdm import tqdm
import tarfile
from MultiDatasetExtractor import MultiDatasetExtractor
import zipfile

class UpdatePatchCreator:

    def __init__(self, topDir, outputDir) -> None:
        self.topDir = topDir
        self.multiDsExtractor = MultiDatasetExtractor(topDir)
        self.multiDsExtractor.scanDir(printScanResult=True)
        self.outputDir = outputDir
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)

    def detectImagesFolderStructure(self, path, acceptExts=['.jpg','.png','.webp','.heic']):
        imageFileList = []
        for ext in acceptExts:
            imageFileList.extend([absPath.relative_to(path) for absPath in pathlib.Path(path).glob('**/*'+ext,case_sensitive=False)])
        return imageFileList


    def detectedNeedUpdatesImageFiles(self,dsDir,backUpDirNameList=['raw_before_sr']):
        originalImageFileList=[]
        backUpImageFileList=[]
        for backUpDirName in backUpDirNameList:
            backUpDir = os.path.join(dsDir,backUpDirName)
            if os.path.isdir(backUpDir):
                originalImageFileList.extend(self.detectImagesFolderStructure(backUpDir))
                backUpImageFileList.extend([pathlib.Path(backUpDirName)/filePath for filePath in originalImageFileList])

        return originalImageFileList,backUpImageFileList



    def generateZip(self, tarName='UpdatePatch', imageInfoFileOnly=False):
        # tar = tarfile.open(os.path.join(self.outputDir, tarName + ".tar.gz"),
        #                    "w:gz")
        with zipfile.ZipFile(os.path.join(self.outputDir, tarName + '.zip'),'w') as myzip:
            for dsDir in self.multiDsExtractor.dirsHasImageInfoJson:
                dsImageInfoPath =  pathlib.Path(dsDir)
                dsDir = dsImageInfoPath.parents[0]
                originalImageFileList,backUpImageFileList = self.detectedNeedUpdatesImageFiles(dsDir)
                updateImageFileList= originalImageFileList+backUpImageFileList
                for updateImageFile in tqdm(updateImageFileList):
                    updateImageFileAbsPath = dsDir/updateImageFile
                    arcnamePath = updateImageFileAbsPath.relative_to(self.topDir)
                    # info = tarfile.TarFile.gettarinfo(tar,updateImageFileAbsPath,arcname=str(arcnamePath))
                    # with open(updateImageFileAbsPath,'rb') as f:
                    #     tar.addfile(info, fileobj=f)
                    try:
                        myzip.write(updateImageFileAbsPath,arcname=arcnamePath,compress_type=zipfile.ZIP_DEFLATED)
                    except FileNotFoundError:
                        print('Missing file:',updateImageFileAbsPath)

                # backup ImageInfo files
                #info = tarfile.TarFile.gettarinfo(tar,dsImageInfoPath,arcname=str(dsImageInfoPath.relative_to(self.topDir)))
                with open(dsImageInfoPath,'rb') as f:
                    myzip.write(dsImageInfoPath,arcname=str(dsImageInfoPath.relative_to(self.topDir)),compress_type=zipfile.ZIP_DEFLATED)
        #tar.close()



if __name__ == '__main__':
    dsc = UpdatePatchCreator('xxx',
                         '.')
    dsc.generateZip()
