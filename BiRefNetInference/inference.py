from __future__ import annotations
import os

import PIL.Image
import torch
import torch
from torchvision import transforms
# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
from pillow_heif import register_heif_opener
register_heif_opener()
class Predictor():
    def __init__(self, weightsDir='.', device='cuda:6') -> None:
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_HR-matting',
                                                trust_remote_code=True,
                                                cache_dir=weightsDir,
                                                local_files_only=False,
                                                device_map=self.device).half()
        self.model.eval()

    def predict(self, raw_image,returnMaskedImage=True):

        image = raw_image
        # Data settings
        image_size = (2048, 2048)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0).to(self.device).half()

        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        if returnMaskedImage:
            image.putalpha(mask)
        return image, mask


if __name__ == '__main__':
    pr = Predictor(weightsDir='./DLToolWeights',
                   device='cuda:0')
    img = PIL.Image.open('15.jpg').convert('RGB')
    pr.predict(img)[0].save('alp.png')