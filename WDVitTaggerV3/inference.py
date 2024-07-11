import os
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F


# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"

# Dataset v2 series of models:
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

# https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/a9eacb1eff904552d3012babfa28b57e1d3e295c/tagger/ui.py#L368
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class FakeHFCacheContext:
    def __init__(self, fakeHome):

        self.home = os.environ.get("HF_HUB_CACHE")
        self.fakeHome = fakeHome

    def __enter__(self):
        os.environ["HF_HUB_CACHE"] = self.fakeHome
        return self

    def __exit__(self, type, value, traceback):
        if self.fakeHome is not None and self.home is not None:
            os.environ["HF_HUB_CACHE"] = self.home


# def pil_pad_square(image: Image.Image) -> Image.Image:
#     w, h = image.size
#     # get the largest dimension so we can pad to a square
#     px = max(image.size)
#     # pad to square with white background
#     canvas = Image.new("RGB", (px, px), (255, 255, 255))
#     canvas.paste(image, ((px - w) // 2, (px - h) // 2))
#     return canvas

def rgbToBgr(inputs):

    return inputs[[2, 1, 0]]


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


class Predictor():
    def __init__(self, weightsDir='.', device='cuda') -> None:
        self.device = device
        with FakeHFCacheContext(weightsDir):
            import timm
            from timm.data import create_transform, resolve_data_config
            from torchvision.transforms import Lambda
            self.model: nn.Module = timm.create_model(
                'hf-hub:SmilingWolf/wd-vit-tagger-v3').eval()
            state_dict = timm.models.load_state_dict_from_hf(
                'SmilingWolf/wd-vit-tagger-v3')
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)

            print("Loading tag list...")
            self.labels: LabelData = self.load_labels_hf(
                repo_id='SmilingWolf/wd-vit-tagger-v3')

        print("Creating data transform...")
        self.model.pretrained_cfg['crop_mode'] = 'border'
        self.transform = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model))
        self.transform.transforms.append(Lambda(rgbToBgr))

        print("Loading image and preprocessing...")

    def load_labels_hf(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> LabelData:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
        try:
            csv_path = hf_hub_download(
                repo_id=repo_id, filename="selected_tags.csv", revision=revision, token=token
            )
            csv_path = Path(csv_path).resolve()
        except HfHubHTTPError as e:
            raise FileNotFoundError(
                f"selected_tags.csv failed to download from {repo_id}") from e

        df: pd.DataFrame = pd.read_csv(csv_path, usecols=["name", "category"])
        tag_data = LabelData(
            names=df["name"].map(
                lambda x: x.replace("_", " ") if x not in kaomojis else x
            ).tolist(),
            rating=list(np.where(df["category"] == 9)[0]),
            general=list(np.where(df["category"] == 0)[0]),
            character=list(np.where(df["category"] == 4)[0]),
        )

        return tag_data

    def get_tags(
        self,
        probs: Tensor,
        labels: LabelData,
        gen_threshold: float,
        char_threshold: float,
        general_mcut_enabled: bool = True,
        character_mcut_enabled: bool = True
    ):
        # Convert indices+probs to labels
        probs = list(zip(labels.names, probs.numpy()))

        # First 4 labels are actually ratings
        rating_labels = dict([probs[i] for i in labels.rating])

        # General labels, pick any where prediction confidence > threshold
        gen_labels = [probs[i] for i in labels.general]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in gen_labels])
            gen_threshold = mcut_threshold(general_probs)
        gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
        gen_labels = dict(
            sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

        # Character labels, pick any where prediction confidence > threshold
        char_labels = [probs[i] for i in labels.character]
        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in char_labels])
            char_threshold = mcut_threshold(character_probs)
            char_threshold = max(0.15, char_threshold)

        char_labels = dict([x for x in char_labels if x[1] > char_threshold])
        char_labels = dict(
            sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

        # Combine general and character labels, sort by confidence
        combined_names = [x for x in gen_labels]
        combined_names.extend([x for x in char_labels])

        # Convert to a string suitable for use as a training caption
        caption = ",".join(combined_names)
        taglist = caption.replace("_", " ").replace(
            "(", "\(").replace(")", "\)")

        return caption, taglist, rating_labels, char_labels, gen_labels

    def processOutputs(self, outputs):
        outputDictList = []
        for output in outputs:
            caption, taglist, ratings, character, general = self.get_tags(
                probs=output,
                labels=self.labels,
                gen_threshold=0.35,
                char_threshold=0.85,
            )
            outputDictList.append({'DBRU_TAG': caption})
        return outputDictList

    def predict(self, img):
        with torch.inference_mode():
            inputs = self.transform(img).unsqueeze(0).to(self.device)
            # from torchvision.utils import save_image
            # save_image(inputs[0],'b.jpg')
            # run the model
            outputs = self.model.forward(inputs)
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)

            outputs = outputs.to("cpu")
            return self.processOutputs(outputs)[0]

    def predict_batch(self, inputs):
        with torch.inference_mode():
            # run the model
            outputs = self.model.forward(inputs.to(self.device))
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)

            outputs = outputs.to("cpu")
            outputDictList = self.processOutputs(outputs)
            return outputDictList


if __name__ == "__main__":
    pr = Predictor(weightsDir='DLToolWeights')
    with open('a.jpg', 'rb') as f:
        imgs = Image.open(f).convert('RGB')
        pr.predict(imgs)
