import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from GOT.utils.conversation import conv_templates, SeparatorStyle
from GOT.utils.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from GOT.model import *
from GOT.utils.utils import KeywordsStoppingCriteria

from PIL import Image

import os
import requests
from PIL import Image
from io import BytesIO
from GOT.model.plug.blip_process import BlipImageEvalProcessor

from transformers import TextStreamer
import re
from GOT.demo.process_results import punctuation_dict, svg_to_html
import string
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = '<imgpad>'

DEFAULT_IM_START_TOKEN = '<img>'
DEFAULT_IM_END_TOKEN = '</img>'


 
translation_table = str.maketrans(punctuation_dict)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    model = GOTQwenForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=151643).eval()

    

    model.to(device='cuda',  dtype=torch.bfloat16)


    # TODO vary old codes, NEED del 
    image_processor = BlipImageEvalProcessor(image_size=1024)

    image_processor_high =  BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

    image = load_image(args.image_file)

    w, h = image.size
    # print(image.size)
    
    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if args.box:
        bbox = eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
            bbox[2] = int(bbox[2]/w*1000)
            bbox[3] = int(bbox[3]/h*1000)
        if args.type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if args.color:
        if args.type == 'format':
            qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs



    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # print(prompt)


    inputs = tokenizer([prompt])


    # vary old codes, no use
    image_1 = image.copy()
    image_tensor = image_processor(image)


    image_tensor_1 = image_processor_high(image_1)


    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            do_sample=False,
            num_beams = 1,
            no_repeat_ngram_size = 20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
        

        if args.render:
            print('==============rendering===============')

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            if '**kern' in outputs:
                import verovio
                from cairosvg import svg2png
                import cv2
                import numpy as np
                tk = verovio.toolkit()
                tk.loadData(outputs)
                tk.setOptions({"pageWidth": 2100, "footer": 'none',
               'barLineWidth': 0.5, 'beamMaxSlope': 15,
               'staffLineWidth': 0.2, 'spacingStaff': 6})
                tk.getPageCount()
                svg = tk.renderToSVG()
                svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")

                svg_to_html(svg, "./results/demo.html")

            if args.type == 'format' and '**kern' not in outputs:

                
                if  '\\begin{tikzpicture}' not in outputs:
                    html_path = "./render_tools/" + "/content-mmd-to-html.html"
                    html_path_2 = "./results/demo.html"
                    right_num = outputs.count('\\right')
                    left_num = outputs.count('\left')

                    if right_num != left_num:
                        outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace('\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace('\\right|', '|').replace('\left.', '.').replace('\\right.', '.')


                    outputs = outputs.replace('"', '``').replace('$', '')

                    outputs_list = outputs.split('\n')
                    gt= ''
                    for out in outputs_list:
                        gt +=  '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n' 
                    
                    gt = gt[:-2]

                    with open(html_path, 'r') as web_f:
                        lines = web_f.read()
                        lines = lines.split("const text =")
                        new_web = lines[0] + 'const text ='  + gt  + lines[1]
                else:
                    html_path = "./render_tools/" + "/tikz.html"
                    html_path_2 = "./results/demo.html"
                    outputs = outputs.translate(translation_table)
                    outputs_list = outputs.split('\n')
                    gt= ''
                    for out in outputs_list:
                        if out:
                            if '\\begin{tikzpicture}' not in out and '\\end{tikzpicture}' not in out:
                                while out[-1] == ' ':
                                    out = out[:-1]
                                    if out is None:
                                        break
    
                                if out:
                                    if out[-1] != ';':
                                        gt += out[:-1] + ';\n'
                                    else:
                                        gt += out + '\n'
                            else:
                                gt += out + '\n'


                    with open(html_path, 'r') as web_f:
                        lines = web_f.read()
                        lines = lines.split("const text =")
                        new_web = lines[0] + gt + lines[1]

                with open(html_path_2, 'w') as web_f_new:
                    web_f_new.write(new_web)


def eval_model_cpu(args):
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # model = GOTQwenForCausalLM.from_pretrained(model_name, 
                                            #    low_cpu_mem_usage=True, 
                                            #    device_map='cuda', 
                                            #    use_safetensors=True, 
                                            #    pad_token_id=151643).eval()
    model = GOTQwenForCausalLM.from_pretrained(model_name, 
                                               low_cpu_mem_usage=False, # 不降低CPU内存使用
                                               device_map=None, # 不指定设备映射
                                               use_safetensors=True, 
                                               pad_token_id=151643).eval()


    
    # model.to(device='cuda',  dtype=torch.bfloat16)
    # 将模型移动到 CPU
    model.to(device='cpu', dtype=torch.float32)  # 使用 float32 类型


    # 创建图像处理器，用于处理不同分辨率的图像
    image_processor = Compose([
        Resize((1024, 1024)),  # 调整图像大小
        ToTensor(),            # 转换为 Tensor
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # TODO vary old codes, NEED del 
    # image_processor = BlipImageEvalProcessor(image_size=1024)

    # image_processor_high =  BlipImageEvalProcessor(image_size=1024)

    use_im_start_end = True

    image_token_len = 256

    image = load_image(args.image_file)

    w, h = image.size
    # print(image.size)
    
    if args.type == 'format':
        qs = 'OCR with format: '
    else:
        qs = 'OCR: '

    if args.box:
        bbox = eval(args.box)
        if len(bbox) == 2:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
        if len(bbox) == 4:
            bbox[0] = int(bbox[0]/w*1000)
            bbox[1] = int(bbox[1]/h*1000)
            bbox[2] = int(bbox[2]/w*1000)
            bbox[3] = int(bbox[3]/h*1000)
        if args.type == 'format':
            qs = str(bbox) + ' ' + 'OCR with format: '
        else:
            qs = str(bbox) + ' ' + 'OCR: '

    if args.color:
        if args.type == 'format':
            qs = '[' + args.color + ']' + ' ' + 'OCR with format: '
        else:
            qs = '[' + args.color + ']' + ' ' + 'OCR: '

    if use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN*image_token_len + DEFAULT_IM_END_TOKEN + '\n' + qs 
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs



    conv_mode = "mpt"
    args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    print(prompt)


    inputs = tokenizer([prompt])


    # vary old codes, no use
    image_1 = image.copy()
    image_tensor = image_processor(image)

    #image_tensor_1 = image_processor_high(image_1)
    image_tensor_1 = image_processor(image_1)

    # input_ids = torch.as_tensor(inputs.input_ids).cuda()
    # 不使用cuda
    input_ids = torch.tensor(inputs.input_ids)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


    # with torch.autocast("cuda", dtype=torch.bfloat16):
    # 关闭梯度计算
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            # images=[(image_tensor.unsqueeze(0).half().cuda(), image_tensor_1.unsqueeze(0).half().cuda())],
            images=[(image_tensor.unsqueeze(0), image_tensor_1.unsqueeze(0))],
            do_sample=False,
            num_beams = 1,
            no_repeat_ngram_size = 20,
            streamer=streamer,
            max_new_tokens=4096,
            stopping_criteria=[stopping_criteria]
            )
        

        if args.render:
            print('==============rendering===============')

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            if '**kern' in outputs:
                import verovio
                from cairosvg import svg2png
                import cv2
                import numpy as np
                tk = verovio.toolkit()
                tk.loadData(outputs)
                tk.setOptions({"pageWidth": 2100, "footer": 'none',
               'barLineWidth': 0.5, 'beamMaxSlope': 15,
               'staffLineWidth': 0.2, 'spacingStaff': 6})
                tk.getPageCount()
                svg = tk.renderToSVG()
                svg = svg.replace("overflow=\"inherit\"", "overflow=\"visible\"")

                svg_to_html(svg, "./results/demo.html")

            if args.type == 'format' and '**kern' not in outputs:

                
                if  '\\begin{tikzpicture}' not in outputs:
                    html_path = "./render_tools/" + "/content-mmd-to-html.html"
                    html_path_2 = "./results/demo.html"
                    right_num = outputs.count('\\right')
                    left_num = outputs.count('\left')

                    if right_num != left_num:
                        outputs = outputs.replace('\left(', '(').replace('\\right)', ')').replace('\left[', '[').replace('\\right]', ']').replace('\left{', '{').replace('\\right}', '}').replace('\left|', '|').replace('\\right|', '|').replace('\left.', '.').replace('\\right.', '.')


                    outputs = outputs.replace('"', '``').replace('$', '')

                    outputs_list = outputs.split('\n')
                    gt= ''
                    for out in outputs_list:
                        gt +=  '"' + out.replace('\\', '\\\\') + r'\n' + '"' + '+' + '\n' 
                    
                    gt = gt[:-2]

                    with open(html_path, 'r') as web_f:
                        lines = web_f.read()
                        lines = lines.split("const text =")
                        new_web = lines[0] + 'const text ='  + gt  + lines[1]
                else:
                    html_path = "./render_tools/" + "/tikz.html"
                    html_path_2 = "./results/demo.html"
                    outputs = outputs.translate(translation_table)
                    outputs_list = outputs.split('\n')
                    gt= ''
                    for out in outputs_list:
                        if out:
                            if '\\begin{tikzpicture}' not in out and '\\end{tikzpicture}' not in out:
                                while out[-1] == ' ':
                                    out = out[:-1]
                                    if out is None:
                                        break
    
                                if out:
                                    if out[-1] != ';':
                                        gt += out[:-1] + ';\n'
                                    else:
                                        gt += out + '\n'
                            else:
                                gt += out + '\n'


                    with open(html_path, 'r') as web_f:
                        lines = web_f.read()
                        lines = lines.split("const text =")
                        new_web = lines[0] + gt + lines[1]

                with open(html_path_2, 'w', encoding='utf-8') as web_f_new:
                    web_f_new.write(new_web)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--box", type=str, default= '')
    parser.add_argument("--color", type=str, default= '')
    parser.add_argument("--render", action='store_true')
    args = parser.parse_args()

    # eval_model(args)
    eval_model_cpu(args)

    # 命令行指令 "D:\Works\OCR\GOT-OCR\GOT-OCR2.0\GOT-OCR-2.0-master\GOT\demo\run_ocr_2.0.py"
    # python D:/Works/OCR/GOT-OCR/GOT-OCR2.0/GOT-OCR-2.0-master/GOT/demo/run_ocr_2.0.py  --model-name  D:/Works/OCR/GOT-OCR/data/GOT_weights/  --image-file  D:/Works/OCR/GOT-OCR/data/image/2.jpg  --type ocr
