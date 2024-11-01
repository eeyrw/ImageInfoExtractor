import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import argparse
import torch
from tqdm import tqdm
import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from typing import Dict, List, Optional, Union
from PIL import Image
import PIL.Image


class LlaVaProcessor:
    def __init__(self, tokenizer, image_processor, mm_use_im_start_end):
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = "llava_v1"

    def format_text(self, text: str):
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text

    def load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

    def get_processed_tokens(self, text: str, image_path: str):
        prompt = self.format_text(text)
        image = self.load_image(image_path)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return image_tensor, input_ids

    def get_processed_tokens_batch(self, batch_text: List[str], image_paths: List[str]):
        prompt = [self.format_text(text) for text in batch_text]
        images = [self.load_image(image_path) for image_path in image_paths]

        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
        ]

        # Determine the maximum length of input_ids in the batch
        max_len = max([len(seq) for seq in batch_input_ids])
        # Pad each sequence in input_ids to the max_len
        padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids)

        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return batch_image_tensor, batch_input_ids
    
class Predictor():
    def __init__(self, weightsDir='.') -> None:
        # Model
        disable_torch_init()
        model_path = os.path.join(weightsDir,'llava-v1.5-13b')
        self.model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(
            model_path,
              None, 
              self.model_name,
              #device = 'cuda:1',
              load_4bit=True)
        self.model.eval()

    def predict(self, raw_image):
        prompts = 'Describe the image as detail as possible.No more than 70 words.'
        prompts2 = 'Describe the image as detail as possible.No more than 80 words.Please describe art style'
        return [{'caption': caption, 'rank': 0, 'isCustomCap': False} for caption in self.generate_caption(
            [raw_image,raw_image],
            [prompts,prompts2])]

    def generate_caption(
            self,
        images: List[PIL.Image.Image],
        querys: List[str],
        conv_mode=None,
        temperature: float = 0.2,
    ) -> str:
        promptList = []
        stopping_criteria = []
        input_idsList=[]
        for qs in querys:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if 'llama-2' in self.model_name.lower():
                conv_mode = "llava_llama_2"
            elif "v1" in self.model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in self.model_name.lower():
                conv_mode = "mpt"
            else:
                conv_mode = "llava_v0"

            if conv_mode is not None and conv_mode != conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, self.conv_mode, self.conv_mode))
            else:
                conv_mode = conv_mode

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            promptList.append(prompt)

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_idsList.append(input_ids)
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria.append(KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids.unsqueeze(0)))
        
        image_tensor = self.image_processor.preprocess(images, return_tensors='pt')['pixel_values']
        image_tensor = image_tensor.half().cuda()

        maxLen = 0
        for input_ids in input_idsList:
            if len(input_ids)>maxLen:
                maxLen = len(input_ids)

        input_ids_pad = []
        for input_ids in input_idsList:
            pad_tensor = torch.tensor(self.tokenizer.pad_token_id,dtype=torch.long).expand(maxLen-len(input_ids))
            input_ids_pad.append(torch.concat((input_ids,pad_tensor)))

        input_ids_pad = torch.stack(input_ids_pad).cuda()





        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids_pad,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=stopping_criteria)

        input_token_len = input_ids_pad.shape[1]
        n_diff_input_output = (input_ids_pad != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')


        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs_texts = []
        for output in outputs:
            output = output.strip()
            if output.endswith(stop_str):
                output = output[:-len(stop_str)]
            output = output.strip()
            outputs_texts.append(output)
        
        return outputs_texts


# def eval_model(args):
#     # Model
#     disable_torch_init()

#     model_name = get_model_name_from_path(args.model_path)
#     tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

#     qs = args.query
#     if model.config.mm_use_im_start_end:
#         qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
#     else:
#         qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

#     if 'llama-2' in model_name.lower():
#         conv_mode = "llava_llama_2"
#     elif "v1" in model_name.lower():
#         conv_mode = "llava_v1"
#     elif "mpt" in model_name.lower():
#         conv_mode = "mpt"
#     else:
#         conv_mode = "llava_v0"

#     if args.conv_mode is not None and conv_mode != args.conv_mode:
#         print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
#     else:
#         args.conv_mode = conv_mode

#     conv = conv_templates[args.conv_mode].copy()
#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

#     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

#     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#     with torch.inference_mode():
#         output_ids = model.generate(
#             input_ids,
#             images=image_tensor,
#             do_sample=True,
#             temperature=0.2,
#             max_new_tokens=1024,
#             use_cache=True,
#             stopping_criteria=[stopping_criteria])

#     input_token_len = input_ids.shape[1]
#     n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
#     if n_diff_input_output > 0:
#         print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
#     outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
#     outputs = outputs.strip()
#     if outputs.endswith(stop_str):
#         outputs = outputs[:-len(stop_str)]
#     outputs = outputs.strip()
#     print(outputs)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="/large_tmp/llava-v1.5-13b")
#     parser.add_argument("--model-base", type=str, default=None)
#     parser.add_argument("--image-file", type=str, required=False, default='/home/songfuqiang/data/sj_small/deviantart_727184218_Hunks of the week #61.webp')
#     parser.add_argument("--query", type=str, required=False, default='Describe image')
#     parser.add_argument("--conv-mode", type=str, default=None)
#     args = parser.parse_args()

#     eval_model(args)

if __name__ == "__main__":
    pred = Predictor('/large_tmp/')
    for i in tqdm(range(1000)):
        print(pred.predict(Image.open('/home/songfuqiang/data/sj_small/deviantart_727184218_Hunks of the week #61.webp')))