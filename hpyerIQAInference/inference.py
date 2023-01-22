import torch
import torchvision
from .HyerIQAModel import HyperNet, TargetNet
from PIL import Image
import os


class Predictor():
    def __init__(self, weightsDir='.') -> None:
        self.model_hyper = HyperNet(
            16, 112, 224, 112, 56, 28, 14, 7,
            weights_dir=weightsDir).cuda()
        self.model_hyper.train(False)
        # load our pre-trained model on the koniq-10k dataset
        hyperIQACkptPath = os.path.join(weightsDir, 'koniq_pretrained.pkl')
        self.model_hyper.load_state_dict((torch.load(hyperIQACkptPath)))

        self.transforms = torchvision.transforms.Compose([
            #torchvision.transforms.Resize((512, 384)),
            #torchvision.transforms.Resize((1024, 1024)),
            # torchvision.transforms.Resize(1024),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])
        self.transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))])

    def predict(self, img, repeatTimes=40):
        imgs = []
        for i in range(repeatTimes):
            w, h = img.size
            if min(w, h) < 224:
                imgTrans = self.transforms2(img)
            else:
                imgTrans = self.transforms(img)
            imgs.append(imgTrans)

        imgs_batch = torch.stack(imgs).cuda()

        # 'paras' contains the network weights conveyed to target network
        paras = self.model_hyper(imgs_batch)

        # Building target network
        model_target = TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False

        # Quality prediction
        # 'paras['target_in_vec']' is the input to target net
        pred = model_target(paras['target_in_vec'])

        # quality score ranges from 0-100, a higher score indicates a better quality
        return torch.mean(pred).detach().cpu().item()

    def predict_multiscale(self, img):
        scores = {}
        # scaleTransList = [('Q2048', 80, torchvision.transforms.Resize(2048)),
        #                   ('Q1024', 20, torchvision.transforms.Resize(1024)),
        #                   ('Q512', 5, torchvision.transforms.Resize(512))]
        scaleTransList = [('Q512', 20, torchvision.transforms.Resize(512))]
        for scaleName, reptTimes, trans in scaleTransList:
            imgs = []
            imgResized = trans(img)
            for i in range(reptTimes):
                imgTrans = self.transforms(imgResized)
                imgs.append(imgTrans)
            with torch.cuda.amp.autocast():
                imgs_batch = torch.stack(imgs).cuda()

                # 'paras' contains the network weights conveyed to target network
                paras = self.model_hyper(imgs_batch)

                # Building target network
                model_target = TargetNet(paras).cuda()
                for param in model_target.parameters():
                    param.requires_grad = False

                # Quality prediction
                # 'paras['target_in_vec']' is the input to target net
                pred = model_target(paras['target_in_vec'])

                # quality score ranges from 0-100, a higher score indicates a better quality
                scores[scaleName] = float(
                    '%.3f' % (torch.mean(pred).detach().cpu().item()))
        return scores


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
