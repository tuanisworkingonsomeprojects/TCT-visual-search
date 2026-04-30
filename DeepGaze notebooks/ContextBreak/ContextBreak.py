import os
import sys
import cv2
import time
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import pickle
import random
import copy

from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms

from glob import glob

from utils import *
# from naturaldesign.naturaldesign import NaturalDesign
sys.path.append("..")


target_min = 128
patch_w = 16

class ContextBreak(Dataset):
    
    def __init__(self, info_dir, dataset_dir, search_size, target_size, per_target=True, is_transform=True, invariant=True):

        
        # _____ PREPROCESS METATDATA _____
        info_file_paths = glob(os.path.join(info_dir, '*.txt'))
        
        self.all_info_df = pd.read_csv(info_file_paths[0], sep='\t')
        
        # Transformation for Unix and Linux file path
        self.all_info_df['search_image'] = self.all_info_df['search_image'].map(lambda x: os.path.join('final', x.replace('\\', '/')))
        self.all_info_df['target_image'] = self.all_info_df['target_image'].map(lambda x: x.replace('\\', '/'))
        
        
        for info_file_path in info_file_paths[1:]:
            
            info_df = pd.read_csv(info_file_path, sep='\t')
            
            # Transformation for Unix and Linux file path
            info_df['search_image'] = info_df['search_image'].map(lambda x: os.path.join('final', x.replace('\\', '/')))
            info_df['target_image'] = info_df['target_image'].map(lambda x: x.replace('\\', '/'))
            
            self.all_info_df = pd.concat([self.all_info_df, info_df], ignore_index=True)
            
        
        
        self.dataset_dir = dataset_dir
        # _____ PREPROCESS METATDATA _____
        
        
        self.search_size = search_size
        self.target_size = target_size if not per_target else None
        
        self.per_target   = per_target
        self.is_transform = is_transform
        
        self.search_transform = transforms.Compose([
            transforms.Resize(search_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.target_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if not per_target else None
        
        
    def __len__(self):
        return len(self.all_info_df)


    def __getitem__(self, idx):
        idx_info = self.all_info_df.iloc[idx, :]

        search_file_path = os.path.join(self.dataset_dir, idx_info['search_image'])
        target_file_path = os.path.join(self.dataset_dir, idx_info['target_image'])

        img    = cv2.imread(search_file_path)
        target = cv2.imread(target_file_path)
        
        
        

        img_PIL    = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        target_PIL = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))

        image_height, image_width, _ = img.shape

        x_center, y_center, w, h = idx_info[['xcenter', 'ycenter', 'objwidth', 'objheight']]

        bbox = np.array([
            (x_center-0.5*w) / image_width,
            (y_center-0.5*h) / image_height,
            w                / image_width,
            h                / image_height
        ]).astype(np.int32)

        solution = np.zeros(self.search_size)
        solution[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1

        if self.is_transform:
            # transform search image
            img = self.search_transform(img_PIL)

            # transform target image
            if not self.per_target:
                target = self.target_transform(target_PIL)
            else:
                target_w, target_h = np.array(target_PIL).shape[1], np.array(target_PIL).shape[0]
                target_max = int((target_min * max(target_w, target_h) / min(target_w, target_h)) // patch_w * patch_w)


                if target_w >= target_h:
                    target_w, target_h = target_max, target_min
                else:
                    target_h, target_w = target_max, target_min

                target_transform = transforms.Compose([
                                transforms.Resize((target_h, target_w)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                target = target_transform(target_PIL)

        return img, target, bbox, solution