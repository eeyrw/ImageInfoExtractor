import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
import os
import torch
import numpy as np


class Predictor():
    def __init__(self, weightsDir='.') -> None:
        self.model = timm.create_model(
            'efficientnet_b3a',pretrained=False, num_classes=2)

        self.model.classifier = nn.Sequential(
            # 1536 is the orginal in_features
            nn.Linear(in_features=1536, out_features=625),
            nn.ReLU(),  # ReLu to be the activation function
            nn.Dropout(p=0.3),
            nn.Linear(in_features=625, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=2),
        )
        state_dict = torch.load(os.path.join(weightsDir,'watermark_model_v1.pt'))

        self.model.load_state_dict(state_dict)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

        self.preprocessing = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def predict(self, img):
        watermark_im = self.preprocessing(img).unsqueeze(0)

        with torch.no_grad():
            batch = watermark_im.to('cuda')
            pred = self.model(batch)
            syms = F.softmax(pred, dim=1).detach().cpu().numpy().tolist()
            water_sym, clear_sym = syms[0]
                
        return {'HAS_WATERMARK':water_sym,'NO_WATERMARK':clear_sym}




def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


