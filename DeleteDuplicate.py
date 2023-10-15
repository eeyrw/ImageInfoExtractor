import json
from shutil import copyfile
import os
from tqdm import tqdm


with open(r'E:\HHH2\dup.json') as f:
    dupList = json.load(f)

# dupDict = {}
# for dup1,dup2,sim in tqdm(dupList):
#     dupDict.setdefault(dup1, set()).add(dup2)
#     dupDict.setdefault(dup2, set()).add(dup1)

# for k,v in dupDict.items():
#     if len(v)>1:
#         print(k)


for dup1,dup2,sim in tqdm(dupList):
    try:
        dup1Size = os.stat(dup1)
        dup2Size = os.stat(dup2)
    except Exception as e:
        continue
    if dup1Size>dup2Size:
        dupFilePath = dup2
    else:
        dupFilePath = dup1
    bakDir = os.path.join('E:\HHH2', 'dup_bak')
    bakImagePath = os.path.join(
        bakDir,os.path.basename(dupFilePath))
    if not os.path.exists(bakDir):
        os.makedirs(bakDir)
    copyfile(dupFilePath, bakImagePath)
    os.unlink(dupFilePath)
    