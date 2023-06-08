# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:19:58 2022

@author: mauricio.jurado
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:18:52 2022

@author: mauricio.jurado
"""

import torch
import time
import numpy as np
import cv2
import os
import time
import torch
import argparse
from metrics import *
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ARGUMENTS
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='0' , help='Device for computing. 0 for GPU, cpu for CPU')
    parser.add_argument('--root', type=str, default="data" , help='Root directory where images and yolo labels are, and where results from inference will be saved')
    parser.add_argument('--weights', type=str, default="yolo.pt" , help='Model weights to use for inference')
    parser.add_argument('--threshold', type=float, default=0.3 , help='IoU threshold for mAP')
    parser.add_argument('--confidence', type=float, default=0.5 , help='Minimun confidence to consider an detection as an eye')
    parser.add_argument('--save_images', type=str2bool, default=True, help='Flag to save images with boxes and classes to local')
    parser.add_argument('--check', type=str2bool, default=True, help='Flag to check with current model in production saved on cloud')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def select_device(device='', batch_size=0, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'YOLOv5 🚀 torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(',', '')), \
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    if not newline:
        s = s.rstrip()
    return torch.device('cuda:0' if cuda else 'cpu')

def inference(opt):

    # SELECT GPU 
    device = select_device(opt.device)

    # LOAD YOLO MODEL WITH CUSTOM WEIGHTS FROM OUR PRETRAINED MODEL
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.weights, force_reload=True)

    # CREATE DIRECTORY WHERE INFERENCE RESULTS ARE GOING TO BE SAVED
    path = os.path.join(opt.root,"images")
    save_result_path = os.path.join(opt.root,"results")
    if os.path.isdir(save_result_path) == False:
        os.makedirs(save_result_path)

    # LIST OF ALL IMAGES TO PROCESS
    files = os.listdir(path)
    total_number_frames = int(len(files))
    print("Starting inference of {0} images in {1}".format(total_number_frames,opt.root))
    total_time = 0
    # PROGRESS BAR WITH TQDM
    for i in tqdm(range(total_number_frames)):
        
        image = files[i]
        frame = cv2.imread(os.path.join(path,image))

        # INFERENCE
        start = time.time()
        result = model(frame)
        end = time.time()
        elapsed = (end - start)
        total_time += elapsed

        # SAVING RESULTS IN A TXT
        name = image.split(".")[0] + ".txt"
        txt_file = os.path.join(save_result_path,name)
        f = open(txt_file, 'w') 
        df = result.pandas().xyxy[0]

        # DATA TO SAVE IN TXT: X, Y, CONFIDENCE AND CLASS
        columns = "xmin, ymin, xmax, ymax, confidence, class\n"
        f.write(columns)
        for j in range(len(df)):
            line = str(df["xmin"][j]) + ", " + str(df["ymin"][j]) + ", " + str(df["xmax"][j]) + ", " + str(df["ymax"][j]) + ", " + str(df["confidence"][j]) + ", " + str(df["class"][j]) + "\n"
            f.write(line)
        f.close()

    fps = total_number_frames / total_time
    return fps

def main(opt):

    fps = inference(opt)
    metrics(opt, fps)

if __name__ == '__main__':
    
    opt = parse_opt()
    main(opt)