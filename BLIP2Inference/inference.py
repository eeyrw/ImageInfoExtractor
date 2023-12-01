# from PIL import Image
# import requests
# from transformers import Blip2Processor, Blip2Model
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b",cache_dir = 'DLToolWeights',local_files_only=True)
# model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16,cache_dir = 'DLToolWeights',local_files_only=True)
# model.to(device)
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# prompt = "Describe image"
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

# outputs = model(**inputs)
# print(processor.decode(outputs[0], skip_special_tokens=True))


from __future__ import annotations
from pillow_heif import register_heif_opener
import os
import string

import gradio as gr
import PIL.Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class Predictor():
    def __init__(self, weightsDir='.', device='cuda') -> None:
        self.device = device
        self.processor = Blip2Processor.from_pretrained(os.path.join(weightsDir, 'blip2-opt-2.7b-fp16'),
                                                       cache_dir=weightsDir, local_files_only=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(os.path.join(weightsDir, 'blip2-opt-2.7b-fp16'),
                                                                   device_map=self.device,
                                                                   torch_dtype=torch.float16,
                                                                   cache_dir=weightsDir,
                                                                   local_files_only=True)

        self.model.eval()

    def predict(self, raw_image):
        return [{'caption': self.generate_caption(raw_image), 'rank': 0, 'isCustomCap': False}]

    def generate_caption(
            self,
        image: PIL.Image.Image,
        decoding_method: str = 'Nucleus sampling',
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        repetition_penalty: float = 1.5,
    ) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.device, torch.float16)
        generated_ids = self.model.generate(
            pixel_values=inputs.pixel_values,
            do_sample=decoding_method == "Nucleus sampling",
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            max_length=70,
            min_length=60,
            num_beams=5,
            top_p=0.9,
        )
        result = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0].strip()
        return result
