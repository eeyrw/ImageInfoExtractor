import os
import imghdr
# Monkeypatch bug in imagehdr
from imghdr import tests
from PIL import Image


def test_jpeg1(h, f):
    """JPEG data in JFIF format"""
    if b'JFIF' in h[:23]:
        return 'jpeg'


JPEG_MARK = b'\xff\xd8\xff\xdb\x00C\x00\x08\x06\x06' \
            b'\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f'


def test_jpeg2(h, f):
    """JPEG with small header"""
    if len(h) >= 32 and 67 == h[5] and h[:32] == JPEG_MARK:
        return 'jpeg'


def test_jpeg3(h, f):
    """JPEG data in JFIF or Exif format"""
    if h[6:10] in (b'JFIF', b'Exif') or h[:2] == b'\xff\xd8':
        return 'jpeg'


tests.append(test_jpeg1)
tests.append(test_jpeg2)
tests.append(test_jpeg3)


def CorrectImageFormat(topDir):
    imagePathList = []
    ext2TypeMap = {
        '.png': 'png',
        '.jpg': 'jpeg',
        '.gif': 'gif',
    }
    type2ExtMap = {
        'png': '.png',
        'jpeg': '.jpg',
        'gif': '.gif',
    }
    for root, dirs, files in os.walk(topDir):
        for filename in files:
            basename, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext == '.png' or ext == '.jpg' or ext == '.gif':
                fullFilePath = os.path.join(root, filename)
                actualType = imghdr.what(fullFilePath)
                if actualType:
                    if ext2TypeMap[ext] != actualType:
                        print('%s is %s' % (filename, actualType))
                        os.rename(fullFilePath, os.path.join(
                            root, basename+type2ExtMap[actualType]))
                else:
                    print('!!!!!!%s is %s' % (filename, actualType))
                    try:
                        os.remove(fullFilePath)
                    except Exception as e:
                        print(e)

    return imagePathList


def ConvertPngToJpg(topDir):
    imagePathList = []
    for root, dirs, files in os.walk(topDir):
        for filename in files:
            basename, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext == '.png':
                fullFilePathSrc = os.path.join(root, filename)
                fullFilePathDst = os.path.join(root, basename+'.jpg')
                img = Image.open(fullFilePathSrc).convert('RGB')
                img.save(fullFilePathDst)  # saving the file on desired path
                os.remove(fullFilePathSrc)
                print('Convert %s to %s' % (fullFilePathSrc, fullFilePathDst))


def CheckFileValidity(topDir, imageInfoList=[]):
    for singleImageInfo in imageInfoList:
        fullFilePath = os.path.join(topDir, singleImageInfo['IMG'])
        if not os.path.isfile(fullFilePath):
            print('Invalid file:%s' % fullFilePath)


# CorrectImageFormat(r"path/to/images")
# ConvertPngToJpg(r"path/to/images")
