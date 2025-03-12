import json
import math
from multiprocessing import Pool
from random import shuffle
from shutil import copyfile
import os
import time
import zipfile
from tqdm import tqdm
from Crypto.Cipher import AES
from Crypto.Util import Padding
import hashlib
import getpass


class ParallelCryptZipper:
    def __init__(self, key):
        self.key = key

    def encrpyt(self, data):
        keyBytes = self.key.encode('utf8')
        keyBytes = Padding.pad(keyBytes, 16, style='pkcs7')
        cipher = AES.new(keyBytes, AES.MODE_OCB)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        assert len(cipher.nonce) == 15
        return tag+cipher.nonce+ciphertext

    def encrpytCopy(self, orinPath, targetPath):
        with open(orinPath, "rb") as fr:
            self.encrpytCopyByData(fr.read(), targetPath)

    def encrpytCopyByData(self, data, targetPath):
        with open(targetPath, "wb") as fw:
            fw.write(self.encrpyt(data))

    def encrpytRead(self, targetPath):
        with open(targetPath, 'rb') as f:
            return self.encrpyt(f.read())

    def zipFunc(self, param):
        zipIdx, subImageInfoList, datasetName, datasetRoot, outputDir = param
        with zipfile.ZipFile(os.path.join(outputDir, f'{datasetName}_{zipIdx}.zip'), 'w') as myzip:
            for singleImageInfo in subImageInfoList:
                orinPath = os.path.join(
                    datasetRoot, singleImageInfo['IMG_ORIGIN'])

                myzip.writestr(os.path.join(datasetName, singleImageInfo['IMG']),
                               self.encrpytRead(orinPath),
                               compress_type=zipfile.ZIP_DEFLATED)

            self.encrpyt(json.dumps(subImageInfoList).encode('utf8'))
            myzip.writestr(os.path.join(datasetName, f'ImageInfo_{zipIdx}.json'),
                           self.encrpyt(json.dumps(
                               subImageInfoList).encode('utf8')),
                           compress_type=zipfile.ZIP_DEFLATED)

    def run(self, subLists, wokerNum):
        with Pool(wokerNum) as p:
            for _ in tqdm(p.imap(self.zipFunc, subLists), total=len(subLists)):
                pass


class CryptImageDsExporter:
    def __init__(self, datasetRoot, datasetName, originImageInfoFile, key, outputDir) -> None:
        self.datasetRoot = datasetRoot
        self.datasetName = datasetName
        self.originImageInfoFile = originImageInfoFile
        self.imageInfoList = []
        self.key = key
        self.outputDir = outputDir
        if not os.path.isdir(self.outputDir):
            os.makedirs(self.outputDir)
        self.imageInfoFilePath = os.path.join(self.outputDir, 'ImageInfo.json')

    def generate(self, splitThresh=10000, wokerNum=20):
        with open(self.originImageInfoFile, 'r') as f:
            self.imageInfoList = json.load(f)
            shuffle(self.imageInfoList)
            self.imageInfoList = [d | {'IMG': hashlib.md5(
                d['IMG'].encode('utf8')).hexdigest()+os.path.splitext(d['IMG'])[1],
                'IMG_ORIGIN': d['IMG']} for d in self.imageInfoList]

            print('Copy and encrypt %d images to %s from %s' %
                  (len(self.imageInfoList), self.outputDir, self.datasetRoot))

            totalImageNum = len(self.imageInfoList)
            imageNumInAZip = splitThresh
            subLists = [(i,
                         self.imageInfoList[i:i+imageNumInAZip],
                         self.datasetName,
                         self.datasetRoot,
                         self.outputDir)
                        for i in range(0, totalImageNum, imageNumInAZip)]

            zipper = ParallelCryptZipper(self.key)
            zipper.run(subLists,wokerNum)

            with zipfile.ZipFile(os.path.join(self.outputDir, f'{self.datasetName}_ImageInfo.zip'), 'w') as myzip:
                zipper.encrpyt(json.dumps(self.imageInfoList).encode('utf8'))
                myzip.writestr('ImageInfo.json', zipper.encrpyt(json.dumps(self.imageInfoList).encode('utf8')),
                               compress_type=zipfile.ZIP_DEFLATED)


if __name__ == '__main__':
    if 'DS_PASSWORD' not in os.environ.keys():
        try:
            key = getpass.getpass()
        except Exception as error:
            print('ERROR', error)
        else:
            print('Password entered:', key)
    else:
        key = os.environ.get['DS_PASSWORD']
        print('Password entered:', key)

    dsc = CryptImageDsExporter('xxxxx',
                               'DiffusionDatasetCrypt',
                               'xxxxxxxxxxx/ImageInfoWeighted.json',
                               key,
                               'xxxxxx')
    dsc.generate()
