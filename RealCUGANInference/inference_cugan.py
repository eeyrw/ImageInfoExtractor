import os
import cv2
from PIL import Image
import numpy
from .upcunet_v3 import RealWaifuUpScaler


class Predictor():
    def __init__(self, weightsDir='.', device='cuda',modeName='up4x-latest-conservative') -> None:
        self.model_name = modeName
        self.device = device

        # determine model paths
        model_path = os.path.join(
            weightsDir, 'CUGAN', self.model_name + '.pth')

        # load+device初始化好当前卡的模型
        self.model = RealWaifuUpScaler(4, model_path, True, self.device)

    def predict(self, img):
        try:
            output = self.model(numpy.array(img), tile_mode=5,
                                cache_mode=0, alpha=1)
        except RuntimeError as error:
            print('Error', error)
            print(
                'If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            # Displaying the converted image
            pil_image = Image.fromarray(output)
            return pil_image
