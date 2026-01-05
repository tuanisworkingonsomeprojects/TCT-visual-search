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

class NaturalDesign(Dataset): 
  def __init__(self, dataset_dir, search_size, target_size, per_target=True, is_transform=True): 
    self.dataset_dir = dataset_dir
    self.search_dir = os.path.join(self.dataset_dir, "stimuli")
    self.target_dir = os.path.join(self.dataset_dir, "target")
    self.bbox_dir = os.path.join(self.dataset_dir, "gt")
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
    return len(os.listdir(self.search_dir))
  
  def __getitem__(self, idx):
    search_file = os.path.join(self.search_dir, 'img{:03}.jpg'.format(idx))
    target_file = os.path.join(self.target_dir, 't{:03}.jpg'.format(idx))
    bbox_file = os.path.join(self.bbox_dir, 'gt{:d}.jpg'.format(idx))
    
    # process search img, target img
    img = Image.open(search_file).convert('RGB')
    target = Image.open(target_file).convert('RGB')
    
    # calculate the bounding box cordinates and the solution image
    gt_img = cv2.imread(bbox_file)[:, :, 0]
    img_width, img_height = gt_img.shape[1], gt_img.shape[0]
    r_bbox, c_bbox = np.where(gt_img == np.max(gt_img))
    xmin, ymin, w, h = c_bbox.min(), r_bbox.min(), c_bbox.max()-c_bbox.min(), r_bbox.max()-r_bbox.min()
    bbox  = np.array([xmin / img_width * self.search_size[1], 
                      ymin / img_height * self.search_size[0], 
                      w / img_width * self.search_size[1],  
                      h / img_height * self.search_size[0]]).astype(np.int32)
    solution = np.zeros(self.search_size)
    solution[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] = 1

    if self.is_transform:
      # transform search image
      img = self.search_transform(img)

      # transform target image
      if not self.per_target:
        target = self.target_transform(target)
      else:
        target_w, target_h = np.array(target).shape[1], np.array(target).shape[0]
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
        target = target_transform(target)

    return img, target, solution, bbox