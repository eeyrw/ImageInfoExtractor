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
from transformers import AutoProcessor, Blip2ForConditionalGeneration

DESCRIPTION = "# [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)"

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



if torch.cuda.is_available():
    processor = AutoProcessor.from_pretrained('DLToolWeights/blip2-opt-2.7b-fp16',cache_dir = 'DLToolWeights',local_files_only=True)
    model = Blip2ForConditionalGeneration.from_pretrained('DLToolWeights/blip2-opt-2.7b-fp16', device_map="auto", torch_dtype=torch.float16,load_in_8bit=False,cache_dir = 'DLToolWeights',local_files_only=True)
else:
    processor = None
    model = None


def generate_caption(
    image: PIL.Image.Image,
    decoding_method: str,
    temperature: float,
    length_penalty: float,
    repetition_penalty: float,
) -> str:
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        pixel_values=inputs.pixel_values,
        do_sample=decoding_method == "Nucleus sampling",
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        max_length=50,
        min_length=20,
        num_beams=5,
        top_p=0.9,
    )
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result


def answer_question(
    image: PIL.Image.Image,
    text: str,
    decoding_method: str,
    temperature: float,
    length_penalty: float,
    repetition_penalty: float,
) -> str:
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        **inputs,
        do_sample=decoding_method == "Nucleus sampling",
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        max_length=30,
        min_length=1,
        num_beams=5,
        top_p=0.9,
    )
    result = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return result


def postprocess_output(output: str) -> str:
    if output and output[-1] not in string.punctuation:
        output += "."
    return output


def chat(
    image: PIL.Image.Image,
    text: str,
    decoding_method: str,
    temperature: float,
    length_penalty: float,
    repetition_penalty: float,
    history_orig: list[str] = [],
    history_qa: list[str] = [],
) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    history_orig.append(text)
    text_qa = f"Question: {text} Answer:"
    history_qa.append(text_qa)
    prompt = " ".join(history_qa)

    output = answer_question(
        image,
        prompt,
        decoding_method,
        temperature,
        length_penalty,
        repetition_penalty,
    )
    output = postprocess_output(output)
    history_orig.append(output)
    history_qa.append(output)

    chat_val = list(zip(history_orig[0::2], history_orig[1::2]))
    return chat_val, history_orig, history_qa


examples = [
    [
        "images/house.png",
        "How could someone get out of the house?",
    ],
    [
        "images/flower.jpg",
        "What is this flower and where is it's origin?",
    ],
    [
        "images/pizza.jpg",
        "What are steps to cook it?",
    ],
    [
        "images/sunset.jpg",
        "Here is a romantic message going along the photo:",
    ],
    [
        "images/forbidden_city.webp",
        "In what dynasties was this place built?",
    ],
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )

    with gr.Box():
        image = gr.Image(type="pil")
        with gr.Tabs():
            with gr.Tab(label="Image Captioning"):
                caption_button = gr.Button("Caption it!")
                caption_output = gr.Textbox(label="Caption Output", show_label=False, container=False)
            with gr.Tab(label="Visual Question Answering"):
                chatbot = gr.Chatbot(label="VQA Chat", show_label=False)
                history_orig = gr.State(value=[])
                history_qa = gr.State(value=[])
                vqa_input = gr.Text(label="Chat Input", show_label=False, max_lines=1, container=False)
                with gr.Row():
                    clear_chat_button = gr.Button("Clear")
                    chat_button = gr.Button("Submit", variant="primary")
    with gr.Accordion(label="Advanced settings", open=False):
        sampling_method = gr.Radio(
            label="Text Decoding Method",
            choices=["Beam search", "Nucleus sampling"],
            value="Nucleus sampling",
        )
        temperature = gr.Slider(
            label="Temperature",
            info="Used with nucleus sampling.",
            minimum=0.5,
            maximum=1.0,
            value=1.0,
            step=0.1,
        )
        length_penalty = gr.Slider(
            label="Length Penalty",
            info="Set to larger for longer sequence, used with beam search.",
            minimum=-1.0,
            maximum=2.0,
            value=1.0,
            step=0.2,
        )
        rep_penalty = gr.Slider(
            label="Repeat Penalty",
            info="Larger value prevents repetition.",
            minimum=1.0,
            maximum=5.0,
            value=1.5,
            step=0.5,
        )

    # gr.Examples(
    #     examples=examples,
    #     inputs=[image, vqa_input],
    # )

    caption_button.click(
        fn=generate_caption,
        inputs=[
            image,
            sampling_method,
            temperature,
            length_penalty,
            rep_penalty,
        ],
        outputs=caption_output,
        api_name="caption",
    )

    chat_inputs = [
        image,
        vqa_input,
        sampling_method,
        temperature,
        length_penalty,
        rep_penalty,
        history_orig,
        history_qa,
    ]
    chat_outputs = [
        chatbot,
        history_orig,
        history_qa,
    ]
    vqa_input.submit(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
    ).success(
        fn=lambda: "",
        outputs=vqa_input,
        queue=False,
        api_name=False,
    )
    chat_button.click(
        fn=chat,
        inputs=chat_inputs,
        outputs=chat_outputs,
        api_name="chat",
    ).success(
        fn=lambda: "",
        outputs=vqa_input,
        queue=False,
        api_name=False,
    )
    clear_chat_button.click(
        fn=lambda: ("", [], [], []),
        inputs=None,
        outputs=[
            vqa_input,
            chatbot,
            history_orig,
            history_qa,
        ],
        queue=False,
        api_name="clear",
    )
    image.change(
        fn=lambda: ("", [], [], []),
        inputs=None,
        outputs=[
            caption_output,
            chatbot,
            history_orig,
            history_qa,
        ],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
