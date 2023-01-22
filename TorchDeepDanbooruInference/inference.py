import numpy as np
import torch
from .deep_danbooru_model import DeepDanbooruModel
import os


class Predictor():
    def __init__(self, weightsDir='.') -> None:
        weightsPath = os.path.join(
            weightsDir, './model-resnet_custom_v3.pt')
        self.model = DeepDanbooruModel()
        self.model.load_state_dict(torch.load(weightsPath))
        self.model.eval()
        self.model.half()
        self.model.cuda()

    def predict(self, img):
        pic = img.resize((512, 512))
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255
        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(a).cuda()
            # first run
            y = self.model(x)[0].detach().cpu().numpy()

        cap = ''
        for i, p in enumerate(y):
            if p >= 0.7:
                cap = cap + self.model.tags[i]+','
        return [{'caption': cap, 'rank': 0, 'isCustomCap': False}]
