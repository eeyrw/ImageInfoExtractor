from .models.blip import blip_decoder
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import clip
import os



class Predictor():
    def __init__(self, weightsDir='.', device='cuda',customCaptionPool=None) -> None:
        image_size = 384
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth'
        med_config_path = os.path.join(
            os.path.dirname(__file__), 'configs/med_config.json')
        self.model = blip_decoder(
            pretrained=model_url, image_size=384, vit='large',
            med_config=med_config_path,
            weights_dir=weightsDir
        )
        self.model.eval()
        self.model = self.model.to(self.device,dtype=torch.float16)

        # self.model_clip, self.preprocess_clip = clip.load(
        #     'ViT-L/14@336px', device=self.device, jit=False, download_root=weightsDir)
        self.custom_texts = customCaptionPool
        if customCaptionPool:
            self.setCustomCaptionCandidates(customCaptionPool)

    def setCustomCaptionCandidates(self, texts):
        self.custom_texts = texts
        texts = clip.tokenize(texts).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.custom_text_features = self.model_clip.encode_text(texts)

    def _filterByCLIP(self, img, texts, topK=3):
        image = self.preprocess_clip(img).unsqueeze(0).to(self.device)
        text = clip.tokenize(texts,truncate=True).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model_clip.encode_image(image)
            text_features = self.model_clip.encode_text(text)
            if self.custom_texts:
                text_features = torch.cat(
                    (text_features, self.custom_text_features))
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @
                          text_features.T).softmax(dim=-1)
            text_probs = text_probs.squeeze(0)

        oringinalTextsLen = len(texts)
        if self.custom_texts:
            texts = texts+self.custom_texts
        argIndex = torch.argsort(text_probs, descending=True)

        finalOut = []
        for rank, i in enumerate(argIndex[0:min(topK, len(argIndex))]):
            isCustomCap = False
            if i >= oringinalTextsLen:
                isCustomCap = True
            if not isCustomCap:
                finalOut.append(
                    {'caption': texts[i], 'rank': rank, 'isCustomCap': isCustomCap})
            elif rank < 4:
                finalOut.append(
                    {'caption': texts[i], 'rank': rank, 'isCustomCap': isCustomCap})
        return finalOut

    def _make_square(self, im, min_size=384, fill_color=(0, 0, 0, 0)):
        x, y = im.size
        size = max(min_size, x, y)
        new_im = Image.new('RGB', (size, size), fill_color)
        new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
        return new_im

    def _inference(self, raw_image, model_n, question, strategy):
        raw_image = self._make_square(raw_image)
        if model_n == 'Image Captioning':
            image = self.transform(raw_image).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                if strategy == "Beam search":
                    caption = self.model.generate(
                        image, sample=False, num_beams=10, num_return_sequences=10, max_length=70, min_length=50)
                else:
                    caption = self.model.generate(
                        image, sample=True, top_p=0.9, num_return_sequences=10, max_length=70, min_length=50)
                return self._filterByCLIP(raw_image, caption)

        else:
            image_vq = self.transform_vq(raw_image).unsqueeze(0).to(self.device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                answer = self.model_vq(
                    image_vq, question, train=False, inference='generate')
            return 'answer: '+answer[0]

    def predict(self, raw_image):
        return self._inference(raw_image, 'Image Captioning', '', 'Nucleus sampling')
    
    def predict_batch(self,imgs):
        imgs = imgs.to(self.device,dtype=torch.float16)
        batchsize = imgs.shape[0]
        numOfSentence = 5
        with torch.no_grad():
            captions = self.model.generate(
                    imgs, sample=True, top_p=0.9, num_return_sequences=numOfSentence, max_length=70, min_length=50)
            
        return [{'CAP':captions[i*numOfSentence:numOfSentence]} for i in range(batchsize)]
