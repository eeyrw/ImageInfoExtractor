from PIL import Image
import pathlib
import argparse
from shutil import copyfile
import os
import sys
import json
sys.path.append("./BLIP")  # "/home/xx/xx" 为工具脚本的存放路径
from BLIP.predict_simple import Predictor


def ImageCaption(topDir, imageInfoList=[]):
    filterList = [singleImageInfo['IMG'] for singleImageInfo in imageInfoList]
    imageCaptionPredictor = Predictor()
    for root, dirs, files in os.walk(topDir):
        for filename in files:
            basename, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext == '.jpg':
                fullFilePath = os.path.join(root, filename)
                relative_path = pathlib.Path(
                    os.path.relpath(fullFilePath, topDir)).as_posix()
                if relative_path in filterList:
                    continue
                with open(fullFilePath, 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                width, height = img.size
                caption = imageCaptionPredictor.predict(img)[0]
                print('%s: %.2f' % (fullFilePath, caption))
                imageInfoList.append(
                    {'IMG': relative_path, 'CAP': caption, 'W': width, 'H': height})

    for singleImageInfo in imageInfoList:
        if 'CAP' not in singleImageInfo.keys():
            fullFilePath = os.path.join(topDir, singleImageInfo['IMG'])
            with open(fullFilePath, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            width, height = img.size
            caption = imageCaptionPredictor.predict(img)[0]
            print('%s: %s' % (fullFilePath, caption))
            singleImageInfo['CAP'] = caption

    return imageInfoList


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsDir', type=str,
                        default=r'F:\NSFW_DS\twitter_nsfw_data')
    parser.add_argument('--jsonPath', type=str,
                        default=r'F:\NSFW_DS\twitter_nsfw_data\ImageInfo.json')
    config = parser.parse_args()
    with open(config.jsonPath, 'r') as f:
        orinImageInfoList = json.load(f)
    imageInfoList = ImageCaption(config.dsDir, orinImageInfoList)
    outputJson = os.path.join(config.dsDir, 'ImageInfo.json')
    with open(outputJson, 'w') as f:
        json.dump(imageInfoList, f)
