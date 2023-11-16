import os
import torch
from os.path import join
from PIL import Image
from .models.dat import DAT
import yaml
from torchvision import transforms
import numpy as np



class Predictor():
    def __init__(self, weightsDir='.') -> None:
        self.device = 'cuda'
        ck = torch.load(os.path.join(weightsDir,'AVA_AOT_vacc_0.8259_srcc_0.7596_vlcc_0.7710.pth'), map_location=torch.device('cpu'))

        with open('./EATInference/configs/dat_base.yaml') as f:
          conf = yaml.load(f, Loader=yaml.FullLoader)['MODEL']['DAT']

        self.model = DAT(**conf)
        self.model.load_state_dict(ck)
        self.model.eval()
        self.model.to(self.device)
        IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
        IMAGE_NET_STD = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])

    def predict(self, img):
        img = img.resize((224, 224))
        img = self.transform(img)
        # 参数是一个图片的数组， unsqueeze相当于创建一个只有一个图片的数组
        img = img.unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
          pred, _, _ = self.model(img)

        pred = self.get_score(pred)
        return {'A_EAT':pred}


    def get_score(self, y_pred):
      w = torch.from_numpy(np.linspace(1, 10, 10))
      w = w.type(torch.FloatTensor)
      w = w.to(self.device)

      w_batch = w.repeat(y_pred.size(0), 1)

      score = (y_pred * w_batch).sum(dim=1)
      score_np = score.data.cpu().numpy()
      return float('%.3f'%score_np[0])



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
