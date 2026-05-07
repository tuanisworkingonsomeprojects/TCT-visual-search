#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import sys
import cv2

import time
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import pickle
# import pickle5 as pickle5
import random
from random import sample
import copy

import os
import shutil
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, normalize
from torchvision import transforms

from utils import *
from ContextBreak.ContextBreak import ContextBreak
sys.path.append("..")


# In[2]:


# ________ ORIGINAL CODE ________
dataset_config = dict(
    dataset_dir = '../datasets/ContextBreak/ooc/ooc_dataset/',
    info_dir    = '../datasets/ContextBreak/ooc/csv_for_exp/',
    search_size = (320, 512),
    target_size = (0, 0),
    per_target   = True,
    is_transform = True
)
# ________ ORIGINAL CODE ________



# ________ MODIFIED CODE ________
# context_dir = '../../../SCEGRAM/01scenes/01object_present'
# target_dir = '../../../SCEGRAM/invariant_objects'
# info_dir = '../../../SCEGRAM/SCEGRAM_Database_scenes_objects.xlsx'
# ________ MODIFIED CODE ________




# context_size, target_size = (320, 512), (128, 128)
dataset = ContextBreak(**dataset_config)


# In[3]:


# define IVSN model
class IVSN_sti(nn.Module):
  def __init__(self, model):
      super(IVSN_sti, self).__init__()
      self.features = nn.Sequential(*list(model.children())[0][:30])
      for param in self.features.parameters():
        param.requires_grad_ = False

  def forward(self, x):
      x = self.features(x)
      return x

class IVSN_tg(nn.Module):
  def __init__(self, model):
      super(IVSN_tg, self).__init__()
      self.features = nn.Sequential(*list(model.children())[0][:30])
      self.pool_layer = nn.AdaptiveMaxPool2d((1, 1))
      for param in self.features.parameters():
        param.requires_grad_ = False

  def forward(self, x):
      x = self.features(x)
      x = self.pool_layer(x)
      return x

from torch.nn.modules.conv import Conv2d
ConvSize, NumTemplates, Mylayer = 1, 512, 31
MMconv = Conv2d(NumTemplates, 1, kernel_size = (ConvSize, ConvSize), stride = (1, 1), padding = (1, 1))


# In[4]:


model_vgg = models.vgg16(pretrained=True)
model_ivsn_sti = IVSN_sti(model_vgg)
model_ivsn_tg = IVSN_tg(model_vgg)


# In[5]:


# with open("[SCEGRAM]bin_idxs.pkl", "rb") as tf:
#     # ______ ORIGINAL CODE _______
#     # bin_info = pickle5.load(tf)
#     # ______ ORIGINAL CODE _______




#     # _____ MODIFIED CODE _____
#     bin_info = pickle.load(tf)
#     # _____ MODIFIED CODE _____


# In[6]:


num_pics, size, image_size = len(dataset), 48, (320, 512)
# IVSN_CON_0_25, IVSN_CON_25_50 = [], []
# IVSN_INCON_0_25, IVSN_INCON_25_50 = [], []
scanpath, attention_map = {}, {}
# index of selected images of first two bins
# selected_imgs = bin_info['con_(0, 25]'].tolist() + bin_info['con_(25, 50]'].tolist() + bin_info['incon_(0, 25]'].tolist() + bin_info['incon_(25, 50]'].tolist()

model_ivsn_sti.eval()
model_ivsn_tg.eval()



attention_isvn = None
mask_isvn = None
tg_isvn = None
cont_isvn = None


IVSN_res = []




with torch.no_grad():
    for id in trange(0, num_pics):
        # if id not in selected_imgs:
        #     continue

        context_images, target_images, bbox, category = dataset[id]
        # get attention map from IVSN model
        context_ivsn = context_images.unsqueeze(0)
        target_ivsn = target_images.unsqueeze(0)
        cont_output_ivsn = model_ivsn_sti(context_ivsn)
        tg_output_ivsn = model_ivsn_tg(target_ivsn)
        MMconv.weight = torch.nn.Parameter(tg_output_ivsn)
        attention_IVSN = MMconv.forward(cont_output_ivsn)
        attention_IVSN = attention_IVSN.detach().squeeze(0)

        # calculate the target bounding box
        tg_loc = bbox_cordinates(bbox, image_size[1], image_size[0])

        # process IVSN attention map
        mask_IVSN = transforms.Resize(image_size)(attention_IVSN)
        mask_IVSN = torch.divide(mask_IVSN, mask_IVSN.max())




        # save the attention map
        # attention_map[id] = (copy.deepcopy(mask_IVSN))










        IVSN_num, path = searchProcesswithPath(tg_loc, mask_IVSN, image_size, size)

        scanpath[id] = path

        IVSN_res.append(IVSN_num)

        # print('IVSN_' + str(id) + ': ' + str(IVSN_num), end = '\t')

# IVSN_CON_res = IVSN_CON_0_25 + IVSN_CON_25_50
# IVSN_INCON_res = IVSN_INCON_0_25 + IVSN_INCON_25_50
# IVSN_res = IVSN_CON_res + IVSN_INCON_res


# In[7]:


np.mean(IVSN_res)#, np.mean(IVSN_CON_res), np.mean(IVSN_INCON_res)


# In[8]:


def sampleIncon(incon_bin_result, con_bin_result, times):
    sample_times = times
    nums = len(con_bin_result)
    print(nums)
    res = np.array([0.0] * 25)

    for id in range(sample_times):
        temp = sample(incon_bin_result, nums)
        temp_accu = model_performance(temp, len(temp))
        res += np.array(temp_accu[:25])

    return (res/sample_times).tolist()

def balanced_accu(res_con, res_incon):
    res = []
    for i in range(25):
        res.append((res_con[i]+res_incon[i])/2)

    return res


# In[9]:


# times = 100
# IVSN_CON_accu = model_performance(IVSN_CON_res, len(IVSN_CON_res))
# IVSN_INCON_accu = sampleIncon(IVSN_INCON_res, IVSN_CON_res, times)
# IVSN_accu = balanced_accu(IVSN_CON_accu, IVSN_INCON_accu)
# IVSN_accu[:10]

IVSN_accu = model_performance(IVSN_res, len(IVSN_res))


# In[10]:


# IVSN_CON_accu[:10], IVSN_INCON_accu[:10]


# In[11]:


IVSN_SCEGRAM_res = {}
IVSN_SCEGRAM_res['combined_accu'] = IVSN_accu
# IVSN_SCEGRAM_res['con_accu'] = IVSN_CON_accu
# IVSN_SCEGRAM_res['incon_accu'] = IVSN_INCON_accu
# IVSN_SCEGRAM_res['con_[0,25)'] = IVSN_CON_0_25
# IVSN_SCEGRAM_res['con_[25,50)'] = IVSN_CON_25_50
# IVSN_SCEGRAM_res['incon_[0,25)'] = IVSN_INCON_0_25
# IVSN_SCEGRAM_res['incon_[25,50)'] = IVSN_INCON_25_50
IVSN_SCEGRAM_res['scanpath'] = scanpath
# IVSN_SCEGRAM_res['attention_map'] = attention_map


# In[12]:


with open("../results/ContextBreak/ContextBreak_IVSN_res.pkl", "wb") as tf:
    pickle.dump(IVSN_SCEGRAM_res, tf)


# In[ ]:





# In[ ]:




