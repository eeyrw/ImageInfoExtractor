from __future__ import annotations

import PIL.Image
import torch
from tqdm import tqdm
from transformers import DINOv3ViTImageProcessorFast, DINOv3ViTModel
from pillow_heif import register_heif_opener
register_heif_opener()


class Predictor():
    def __init__(self, weightsDir='.', device='cuda:6') -> None:
        self.device = device
        pretrained_model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
        self.processor = DINOv3ViTImageProcessorFast.from_pretrained(
            pretrained_model_name,
            dtype=torch.bfloat16
            )
        self.model = DINOv3ViTModel.from_pretrained(
            pretrained_model_name,
            device_map=self.device,
            local_files_only=False,
            cache_dir=weightsDir,
            dtype=torch.bfloat16
        )
        self.model.eval()

    def predict(self, raw_image):

        # Data settings
        inputs = self.processor(images=raw_image, return_tensors="pt", size={
                                'width': 512, 'height': 512}).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)

        return {"IMG_EMBD": outputs.pooler_output.squeeze().
                to(device='cpu', dtype=torch.float32).numpy().tolist()}


if __name__ == '__main__':
    pr = Predictor(weightsDir='./DLToolWeights',
                   device='cuda:0')
    img = PIL.Image.open('xxx.heic').convert('RGB')
    # for i in tqdm(range(1000)):
    #     pr.predict(img)
    print(pr.predict(img))
