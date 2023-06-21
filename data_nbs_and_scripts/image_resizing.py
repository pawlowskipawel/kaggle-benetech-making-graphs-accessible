import argparse
import glob
import cv2
import os

import pandas as pd
import numpy as np

import torch.nn.functional as F
import torch
from tqdm import tqdm
from joblib import Parallel, delayed


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--input-dir')
    argument_parser.add_argument('--image-size', nargs="+", type=int)
    argument_parser.add_argument('--letterbox', nargs='?', type=bool, const=True, default=False)
    argument_parser.add_argument('--override', nargs='?', type=bool, const=True, default=False)

    return argument_parser.parse_args()

def letterbox(img, input_shape):
    img_h, img_w = img.shape[:2]            
    new_h, new_w = input_shape[1], input_shape[0] 
    
    if (new_w / img_w) <= (new_h / img_h):      
        new_h = int(img_h * new_w / img_w)  
    else:
        new_w = int(img_w * new_h / img_h)   
         
    resized = cv2.resize(img, (new_w, new_h))
    
    img = np.full((input_shape[1], input_shape[0]), 0, dtype=np.uint8) 
    img[0:new_h, 0:new_w] = resized

    return img

def process(img_path, save_folder="", resize=None, do_letterbox=False):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path, 0)
    
    if do_letterbox:
        img = letterbox(img, resize)
    else:
        img = cv2.resize(img, resize)
        
    cv2.imwrite(os.path.join(save_folder, img_name), img)
    

if __name__ == '__main__':
    args = parse_args()
    image_paths = glob.glob(f"{args.input_dir}/*.jpg")
    SAVE_FOLDER = f"/home/pawel/Projects/benetech-making-graphs-accessible/{args.input_dir}{'_' + 'x'.join([str(s) for s in args.image_size]) if not args.override else ''}{'_LB' if args.letterbox else ''}"

    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
        
    _ = Parallel(n_jobs=14)(
        delayed(process)(uid, save_folder=SAVE_FOLDER, resize=tuple(args.image_size), do_letterbox=args.letterbox)
        for uid in tqdm(image_paths))