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

target_min = 128
patch_w = 16

class coco18(Dataset): 
  def __init__(self, dataset_dir, task, search_size, target_size, per_target=True, is_transform=True): 
    self.dataset_dir = dataset_dir
    self.task = task
    self.search_dir = os.path.join(self.dataset_dir, "images", self.task)
    with open(os.path.join(self.dataset_dir, "image_info.pkl"), 'rb') as handle:
      self.image_info = pickle.load(handle)
    with open(os.path.join(self.dataset_dir, "random_filenames_pairs_0915.pkl"), 'rb') as handle:
      self.search_target_pair = pickle.load(handle)
    self.bbox_info = np.load(os.path.join(self.dataset_dir, "processed/bbox_annos.npy"), allow_pickle=True).item()
    self.search_size = search_size
    self.target_size = target_size if not per_target else None
    self.per_target = per_target
    self.is_transform = is_transform

    self.search_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(search_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    self.target_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]) if not per_target else None

    self.resize_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(search_size),
    ])

  def __len__(self):
    return len(self.image_info[self.task]['names'])
  
  def __getitem__(self, idx):
    search_name = self.image_info[self.task]['names'][idx]
    search_key = self.task + '_' + search_name
    target_name = self.search_target_pair[search_key].split('_')[-1]
    target_key = self.task + '_' + target_name
    search_file = os.path.join(self.search_dir, search_name)
    target_file = os.path.join(self.search_dir, target_name)
    
    # calculate the bounding box cordinates and the solution image
    bbox = self.bbox_info[search_key]
    solution = np.zeros(self.search_size)
    solution[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1

    # process search img
    img = Image.open(search_file).convert('RGB')
    target = Image.open(target_file).convert('RGB')
    t_bbox = self.bbox_info[target_key]

    if self.is_transform:
      # transform search image
      img = self.search_transform(img)
      target = self.resize_transform(target)
      target = target[:, t_bbox[1]:t_bbox[1]+t_bbox[3], t_bbox[0]:t_bbox[0]+t_bbox[2]]

      # transform target image
      if not self.per_target:
        target = self.target_transform(target)
      else:
        target_w, target_h = target.shape[2], target.shape[1]
        target_max = int((target_min * max(target_w, target_h) / min(target_w, target_h)) // patch_w * patch_w)
        if target_w >= target_h:
            target_w, target_h = target_max, target_min
        else:
            target_h, target_w = target_max, target_min
        
        target_transform = transforms.Compose([
                          transforms.Resize((target_h, target_w)),
                          transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        target = target_transform(target)

    return img, target, solution, bbox