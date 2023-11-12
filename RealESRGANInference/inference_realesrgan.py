import os

import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from PIL import Image
from realesrgan import RealESRGANer
import numpy


class Predictor():
    def __init__(self, weightsDir='.', modeName='RealESRGAN_x2plus') -> None:
        """Inference demo for Real-ESRGAN.
        """
        # help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
        #           'realesr-animevideov3 | realesr-general-x4v3')
        self.model_name = modeName
        self.outscale = 2

        # determine models according to model names
        self.model_name = self.model_name .split('.')[0]
        if self.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif self.model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif self.model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif self.model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']

        # determine model paths
        model_path = os.path.join(weightsDir, self.model_name + '.pth')
        if not os.path.isfile(model_path):
            for url in file_url:
                # model_path will be updated
                model_path = load_file_from_url(
                    url=url, model_dir=weightsDir, progress=True, file_name=None)

        # restorer
        self.upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0)

    def predict(self, img):
        try:
            opencvImage = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
            output, _ = self.upsampler.enhance(opencvImage, outscale=self.outscale)
        except RuntimeError as error:
            print('Error', error)
            print(
                'If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            # converted from BGR to RGB
            color_coverted = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # Displaying the converted image
            pil_image = Image.fromarray(color_coverted)
            return pil_image
