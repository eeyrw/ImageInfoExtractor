import torch
import torchvision
from .HyerIQAModel import HyperNet, TargetNet
from PIL import Image
import os
from typing import Tuple, List, Optional
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms.transforms import _setup_size


class BatchRandomCrop(torch.nn.Module):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int], batch: int) -> List[Tuple[int, int, int, int]]:
        """Get parameters for ``crop`` for a batch random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
            batch num (int): How many batch

        Returns:
            List of tuple: params [(i, j, h, w)] to be passed to ``crop`` for random crop.
        """
        _, h, w = F.get_dimensions(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                f"Required crop size {(th, tw)} is larger then input image size {(h, w)}")

        params_list = []

        for b in range(batch):
            if w == tw and h == th:
                params_list.append((0, 0, h, w))
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                params_list.append((i, j, th, tw))
        return params_list

    def __init__(self, size, batch=1, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(
            _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size."))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.batch = batch

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        _, height, width = F.get_dimensions(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        batch_crops = []
        for i, j, h, w in self.get_params(img, self.size, self.batch):
            batch_crops.append(F.crop(img, i, j, h, w))

        return batch_crops

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, padding={self.padding})"


class Predictor():
    def __init__(self, weightsDir='.', device='cuda') -> None:
        self.device = device
        self.model_hyper = HyperNet(
            16, 112, 224, 112, 56, 28, 14, 7,
            weights_dir=weightsDir).to(device)
        self.model_hyper.train(False)
        # load our pre-trained model on the koniq-10k dataset
        hyperIQACkptPath = os.path.join(weightsDir, 'koniq_pretrained.pkl')
        self.model_hyper.load_state_dict((torch.load(hyperIQACkptPath)))

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512),
            BatchRandomCrop(size=224, batch=40),
            torchvision.transforms.Lambda(lambda crops: torch.stack(
                [torchvision.transforms.ToTensor()(crop) for crop in crops])),  # returns a 4D tensor
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])

    def predict(self, imgs):
        imgs = self.transform(imgs)
        scores = {}
        with torch.cuda.amp.autocast():
            if len(imgs.shape) == 4:
                imgs_batch = imgs.to(self.device)
            else:
                raise RuntimeError('Wrong input dimension')

            # 'paras' contains the network weights conveyed to target network
            paras = self.model_hyper(imgs_batch)

            # Building target network
            model_target = TargetNet(paras).to(self.device)
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            # 'paras['target_in_vec']' is the input to target net
            pred = model_target(paras['target_in_vec'])
            # quality score ranges from 0-100, a higher score indicates a better quality
            scores['Q512'] = float(
                '%.3f' % (torch.mean(pred).detach().cpu().item()))
        return scores

    def predict_batch(self, imgs):
        scores_list = []
        with torch.cuda.amp.autocast():
            if len(imgs.shape) == 5:
                bb, b, c, h, w = imgs.shape
                imgs_batch = torch.reshape(imgs, (-1, c, h, w))
                imgs_batch = imgs_batch.to(self.device, dtype=torch.float16)
            else:
                raise RuntimeError('Wrong input dimension')
            # 'paras' contains the network weights conveyed to target network
            paras = self.model_hyper(imgs_batch)

            # Building target network
            model_target = TargetNet(paras).to(self.device)
            for param in model_target.parameters():
                param.requires_grad = False

            # Quality prediction
            # 'paras['target_in_vec']' is the input to target net
            pred = model_target(paras['target_in_vec'])
            pred = torch.reshape(pred, (bb, b))

            # quality score ranges from 0-100, a higher score indicates a better quality
            MOSList = torch.mean(pred, dim=1)
            scores_list = [{'Q512': float(
                '%.3f' % q.detach().cpu().item())} for q in MOSList]
        return scores_list


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
