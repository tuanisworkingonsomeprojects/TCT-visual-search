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

import os
import shutil
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms

sys.path.append("..")

from ..utils import *


target_min = 128
patch_w = 16

class scegram(Dataset): 
    def __init__(self, info_dir, dataset_dir, search_size, target_size, per_target=True, is_transform=True, invariant=True): 
        df_info = pd.read_excel(info_dir, engine='openpyxl')
        self.df_info = df_info[df_info.obj_name != 'XXX']
        self.image_width = self.df_info.iloc[0, :]['sce_width']
        self.image_height = self.df_info.iloc[0, :]['sce_height']
        self.dataset_dir = dataset_dir
        self.search_dir = os.path.join(self.dataset_dir, "01scenes/01object_present")
        if invariant:
            self.target_dir = os.path.join(self.dataset_dir, "invariant_objects")
        else:
            self.target_dir = os.path.join(self.dataset_dir, "02objects")
        self.target_dic = {filename[3:-4]: filename for filename in os.listdir(self.target_dir)}
        self.search_size = search_size
        self.target_size = target_size if not per_target else None
        self.per_target = per_target
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
        return len(self.df_info)
    
    def __getitem__(self, idx):
        idx_info = self.df_info.iloc[idx, :]
        search_file = os.path.join(self.search_dir, idx_info['sce_file_name'])
        target_file = os.path.join(self.target_dir, self.target_dic[idx_info['obj_name']])
        img_category = idx_info['sce_file_name'][:-4].split('_')[-1]

        # calculate the bounding box cordinates and the solution image
        x_center, y_center, w, h = idx_info['obj_x_center'], idx_info['obj_y_center'], idx_info['obj_width'], idx_info['obj_height']
        bbox = np.array([(x_center-0.5*w) / self.image_width * self.search_size[1], 
                        (y_center-0.5*h) / self.image_height * self.search_size[0],
                        w / self.image_width * self.search_size[1], 
                        h / self.image_height * self.search_size[0]]).astype(np.int32)
        solution = np.zeros(self.search_size)
        solution[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1
        
        # process search img, target img
        img = cv2.imread(search_file)
        target = cv2.imread(target_file)
        img_PIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        target_PIL = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))

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

        return img, target, solution, bbox, img_category



    
