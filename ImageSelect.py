import json
import argparse
from shutil import copyfile
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonPath', type=str, default=r'F:\NSFW_DS\MHPHOTOS\ImageInfo.json')
    parser.add_argument('--dsRoot', type=str, default=r'F:\NSFW_DS\MHPHOTOS')
    parser.add_argument('--output', type=str, default=r'F:\NSFW_DS\high_quality')
    parser.add_argument('--threshold', type=float, default=70)
    config = parser.parse_args()

    outputJson = config.jsonPath
    with open(outputJson, 'r') as f:
        imageInfo = json.load(f)
    
    filterList = [singleImageInfo for singleImageInfo in imageInfo if singleImageInfo['Q512']>70 and singleImageInfo['H']*singleImageInfo['W']>700*700]
    for singleImageInfo in tqdm(filterList):
        orinPath = os.path.join(config.dsRoot,singleImageInfo['IMG'])
        fileNameWithExt = os.path.basename(orinPath)
        targetPath = os.path.join(config.output,fileNameWithExt)
        copyfile(orinPath,targetPath)

    
