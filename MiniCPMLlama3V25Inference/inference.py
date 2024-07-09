from __future__ import annotations
import os

import PIL.Image
import torch
from transformers import AutoModel, AutoTokenizer

class Predictor():
    def __init__(self, weightsDir='.', device='cuda:6') -> None:
        self.device = device
        
        self.model = AutoModel.from_pretrained('/home/conti/ImageInfoExtractor/DLToolWeights/MiniCPMLlama3V25',
                                                trust_remote_code=True,
                                                cache_dir=weightsDir, 
                                                local_files_only=True,
                                                device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('/home/conti/ImageInfoExtractor/DLToolWeights/MiniCPMLlama3V25', 
                                                       trust_remote_code=True,
                                                    cache_dir=weightsDir, 
                                                local_files_only=True)

        self.model.eval()

    def predict(self, raw_image):

        image = raw_image
        question = 'Describe the image in detail in 60 words'
        msgs = [{'role': 'user', 'content': question}]

        res = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )
        return [{'caption': res, 'rank': 0, 'isCustomCap': False}]
    

if __name__ == '__main__':
    pr = Predictor(weightsDir='ImageInfoExtractor/DLToolWeights')
    img = PIL.Image.open('/xxx.jpg').convert('RGB')
    for i in range(100):
        print(pr.predict(img))

