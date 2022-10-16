import numpy as np
import matplotlib.pyplot as plt
import json
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonPath', type=str, default=r"F:\NSFW_DS\various_source_nsfw_data\ImageInfo.json")
    config = parser.parse_args()

    outputJson = config.jsonPath
    with open(outputJson, 'r') as f:
        imageInfo = json.load(f)

    imageNum = len(imageInfo)
    print('%s images'%imageNum)
    imageScoreListQ512 = [singleImageInfo['Q512'] for singleImageInfo in imageInfo]
    imageScoreListQ512 = np.array(imageScoreListQ512)
    imageScoreListQ1024 = [singleImageInfo['Q1024'] for singleImageInfo in imageInfo]
    imageScoreListQ1024 = np.array(imageScoreListQ1024)
    imageScoreListQ2048 = [singleImageInfo['Q2048'] for singleImageInfo in imageInfo]
    imageScoreListQ2048 = np.array(imageScoreListQ2048)
    imagePixelList = [singleImageInfo['H']*singleImageInfo['W'] for singleImageInfo in imageInfo]
    imagePixelList = np.array(imagePixelList)
    imagePixelList = np.sqrt(imagePixelList)

    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(3,1,1)
    ax.set_title('PDF of %s' % outputJson)
    ns, edgeBin, patches = ax.hist(np.array(imageScoreListQ512), bins=100, rwidth=0.8,label='Q512')
    ns, edgeBin, patches = ax.hist(np.array(imageScoreListQ1024), bins=100, rwidth=0.8,label='Q1024')
    ns, edgeBin, patches = ax.hist(np.array(imageScoreListQ2048), bins=100, rwidth=0.8,label='Q2048')
    ax.legend(prop={'size': 10})
    ax2 = fig.add_subplot(3,1,2)
    ax2.set_title('CDF of %s' % outputJson)
    ns, edgeBin, patches = ax2.hist(np.array(imageScoreListQ512), bins=100, rwidth=0.8,label='Q512',cumulative=True,histtype='step')
    ns, edgeBin, patches = ax2.hist(np.array(imageScoreListQ1024), bins=100, rwidth=0.8,label='Q1024',cumulative=True,histtype='step')
    ns, edgeBin, patches = ax2.hist(np.array(imageScoreListQ2048), bins=100, rwidth=0.8,label='Q2048',cumulative=True,histtype='step')
    ax2.legend(prop={'size': 10})

    ax3 = fig.add_subplot(3,1,3)
    # ax3.set_title('Pixel Dis of %s' % outputJson)
    # ns, edgeBin, patches = ax3.hist(np.array(imagePixelList), bins=100, rwidth=0.8)
    # ax3.legend(prop={'size': 10})
    ax3.hist2d(imagePixelList, imageScoreListQ512, bins=(200,50))
    plt.show()
