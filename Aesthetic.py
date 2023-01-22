import torch
import torch.nn as nn
import clip


# if you changed the MLP architecture during training, change it also here:
class MLP(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class Predictor():
    def __init__(self, weightsPath="./sac+logos+ava1-l14-linearMSE.pth") -> None:
        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

        # load the model you trained previously or the model available in this repo
        s = torch.load(weightsPath)

        self.model.load_state_dict(s)

        self.model.to("cuda")
        self.model.eval()

        self.model2, self.preprocess = clip.load(
            "ViT-L/14", device="cuda", download_root='./clip_root')  # RN50x64

    def predict(self, img):

        image = self.preprocess(img).unsqueeze(0).to("cuda")

        with torch.no_grad():
            image_features = self.model2.encode_image(image)

        im_emb_arr = normalized(image_features.cpu().detach().numpy())

        prediction = self.model(torch.from_numpy(im_emb_arr).to(
            "cuda").type(torch.cuda.FloatTensor))
        return {'A': float('%.3f' % prediction.detach().cpu()[0][0])}
