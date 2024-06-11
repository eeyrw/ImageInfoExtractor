import os


def DetectImageInfoFolder(path,ImageInfoJsonFileName='ImageInfo.json'):
    dirsHasImageInfoJson = []
    dirsHasNotImageInfoJson = []
    dir_HasImageInfo = {}

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path) and os.path.basename(file_path)==ImageInfoJsonFileName:
            dirsHasImageInfoJson.append(file_path)
            return dirsHasImageInfoJson,dirsHasNotImageInfoJson
        elif os.path.isdir(file_path):
            dirsHasImageInfoJson_,dirsHasNotImageInfoJson_=DetectImageInfoFolder(file_path)
            dirsHasImageInfoJson.extend(dirsHasImageInfoJson_)
            dirsHasNotImageInfoJson.extend(dirsHasNotImageInfoJson_)
            if len(dirsHasImageInfoJson_)>0:
                dir_HasImageInfo[file_path] = True
            else:
                dir_HasImageInfo[file_path] = False

    if len(dirsHasImageInfoJson)>0:
        for path,hasImageInfo in dir_HasImageInfo.items():
            if not hasImageInfo:
                dirsHasNotImageInfoJson.append(path)
          
    return dirsHasImageInfoJson,dirsHasNotImageInfoJson

dirsHasImageInfoJson,dirsHasNotImageInfoJson = DetectImageInfoFolder(r'/large_tmp/DiffusionDataset/')

for dir in dirsHasImageInfoJson:
    print('--',dir)
for dir in dirsHasNotImageInfoJson:
    print('??',dir)
