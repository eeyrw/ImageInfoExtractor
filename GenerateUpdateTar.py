import pathlib
import os
from tqdm import tqdm
import tarfile
from MultiDatasetExtractor import MultiDatasetExtractor

class UpdateTarCreator:

    def __init__(self, topDir, outputDir) -> None:
        self.topDir = topDir
        self.multiDsExtractor = MultiDatasetExtractor(topDir)
        self.multiDsExtractor.scanDir(printScanResult=True)
        self.outputDir = outputDir
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)

    def detectImagesFolderStructure(self, path, acceptExts=['.jpg','.png','.webp','.heic']):
        imageFileList = []

        for entry in os.scandir(path):
            file_path = entry.path
            if entry.is_file(follow_symlinks=False) and pathlib.Path(file_path).suffix.lower() in acceptExts:
                dirRelativepath = pathlib.Path(os.path.relpath(file_path, path))
                imageFileList.append(dirRelativepath)
            elif entry.is_dir(follow_symlinks=False):
                imageFileListNextLevel = self.detectImagesFolderStructure(
                    file_path)
                imageFileList.extend(imageFileListNextLevel)
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



    def generateTar(self, tarName='UpdateTar', imageInfoFileOnly=False):
        tar = tarfile.open(os.path.join(self.outputDir, tarName + ".tar.gz"),
                           "w:gz")
        for dsDir in self.multiDsExtractor.dirsHasImageInfoJson:
            dsDir = pathlib.Path(dsDir).parents[0]
            originalImageFileList,backUpImageFileList = self.detectedNeedUpdatesImageFiles(dsDir)
            updateImageFileList= originalImageFileList+backUpImageFileList
            for updateImageFile in tqdm(updateImageFileList):
                updateImageFileAbsPath = dsDir/updateImageFile
                info = tarfile.TarFile.gettarinfo(tar,updateImageFileAbsPath,arcname=str(dsDir.stem/updateImageFile))
                with open(updateImageFileAbsPath,'rb') as f:
                    tar.addfile(info, fileobj=f)
        tar.close()



if __name__ == '__main__':
    dsc = UpdateTarCreator('xxx',
                         '.')
    dsc.generateTar()
