from __future__ import annotations
import os

import PIL.Image
import torch
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
# Load BiRefNet with weights
from transformers import AutoModelForImageSegmentation
from pillow_heif import register_heif_opener
register_heif_opener()
class Predictor():
    def __init__(self, weightsDir='.', device='cuda:6') -> None:
        self.device = device
        self.model = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet',
                                                trust_remote_code=True,
                                                cache_dir=weightsDir,
                                                local_files_only=True,
                                                device_map=self.device)
        self.model.eval()

    def predict(self, raw_image):

        image = raw_image
        # Data settings
        image_size = (1024, 1024)
        transform_image = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        input_images = transform_image(image).unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        return image, mask


if __name__ == '__main__':
    pr = Predictor(weightsDir='ImageInfoExtractor/DLToolWeights',
                   device='cuda:6')
    img = PIL.Image.open('xxx.heic').convert('RGB')
    pr.predict(img)[0].save('alp.png')