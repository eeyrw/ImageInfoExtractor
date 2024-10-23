from __future__ import annotations
import json
import os

import PIL.Image
import torch
from transformers import AutoModel, AutoTokenizer

class Predictor():
    def __init__(self, weightsDir='.', device='cuda:6') -> None:
        self.device = device
        
        modelPath = os.path.abspath(os.path.join(weightsDir,'MiniCPM-V-2_6-int4'))

        self.model = AutoModel.from_pretrained(modelPath,
                                                trust_remote_code=True,
                                                cache_dir=weightsDir, 
                                                local_files_only=True,
                                                device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(modelPath, 
                                                       trust_remote_code=True,
                                                    cache_dir=weightsDir, 
                                                local_files_only=True)

        self.model.eval()

    def predict(self, raw_image, prompt='Caption the image in detail in 60 words in concise way.'):

        image = raw_image
        question = prompt
        msgs = [{'role': 'user', 'content': [image,question]}]

        res = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=0.7,
            # system_prompt='' # pass system_prompt if needed
        )
        return [{'caption': res, 'rank': 0, 'isCustomCap': False}]
    

if __name__ == '__main__':
    pr = Predictor(weightsDir='DLToolWeights')
    with open('prompt.txt') as f:
        prompt = f.read()
    img = PIL.Image.open('a.jpg').convert('RGB')
    for i in range(100):
        result = pr.predict(img,prompt=prompt)
        info = json.loads(result[0]['caption'])
        print(info)

