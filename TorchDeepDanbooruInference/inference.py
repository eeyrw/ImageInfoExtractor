import numpy as np
import torch
from .deep_danbooru_model import DeepDanbooruModel
import os
from torchvision import transforms

class Predictor():
    def __init__(self, weightsDir='.',device='cuda',threshold=0.7) -> None:
        self.device = device
        self.dtype = torch.float16
        weightsPath = os.path.join(
            weightsDir, './model-resnet_custom_v3.pt')
        self.model = DeepDanbooruModel()
        self.model.load_state_dict(torch.load(weightsPath))
        self.model.eval()
        self.model.to(self.device,dtype=self.dtype)
        self.transform = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()])
        self.threshold = threshold

    def predict(self, img):
        with torch.no_grad():
            x = x.to(self.device,dtype=self.dtype)
            # first run
            y = self.model(x)[0].detach().cpu().numpy()

        cap = ''
        for i, p in enumerate(y):
            if p >= 0.7:
                cap = cap + self.model.tags[i]+','
        return [{'caption': cap, 'rank': 0, 'isCustomCap': False}]
    
    def predict_batch(self, imgs):
        with torch.no_grad():
            xs = imgs.to(self.device,dtype=self.dtype)
            # first run
            ys = self.model(xs).detach().cpu().numpy()

        batch_result = []
        for y in ys:
            cap = ''
            for i, p in enumerate(y):
                if p >= self.threshold:
                    cap = cap + self.model.tags[i]+','
            batch_result.append({'DBRU_TAG':cap})
        return batch_result