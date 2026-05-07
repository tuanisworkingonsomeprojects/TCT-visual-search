#!/usr/bin/env python
# coding: utf-8

# In[1]:


on_colab = False

import os

if not on_colab:
    root_folder = '../..'

store_attn_on_local: bool = False
store_rel_on_local:  bool = True

if store_attn_on_local or store_rel_on_local:
    if on_colab:
        if os.path.exists('/src/TCT-visual-search/results'):
            os.unlink('/src/TCT-visual-search/results')
        os.symlink(os.path.abspath('../results'), '/src/TCT-visual-search/results')

# _________ APPEND SOME PATH TO IMPORT SOME LIBRARY BELOW _____________
import sys

sys.path.append('..')
sys.path.append('../ViT_utils')
# _________ APPEND SOME PATH TO IMPORT SOME LIBRARY BELOW _____________


# In[2]:


import sys
import cv2
import time
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import pickle

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

import matplotlib.pyplot as plt
import numpy as np
from scipy.datasets import face
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)

import deepgaze_pytorch
from deepgaze_pytorch import modules

from utils import *
sys.path.append("..")
from ContextBreak.ContextBreak import ContextBreak


# In[3]:


dataset_config = dict(
    dataset_dir = '../datasets/ContextBreak/ooc/ooc_dataset/',
    info_dir    = '../datasets/ContextBreak/ooc/csv_for_exp/',
    search_size = (320, 512),
    target_size = (128, 128),
    per_target   = False,
    is_transform = True
)




# context_size, target_size = (224, 224), (224, 224)
dataset = ContextBreak(**dataset_config)


# In[4]:


# with open("../IVSN/[SCEGRAM]bin_idxs.pkl", "rb") as tf:
#     bin_info = pickle.load(tf) 


# In[5]:

def worker(gpu_id, dataset, indices, result_dict):

    import torch

    import deepgaze_pytorch

    from torchvision import transforms

    from scipy.ndimage import zoom

    from scipy.special import logsumexp

    import numpy as np

    import random

    device = torch.device(f"cuda:{gpu_id}")

    torch.cuda.set_device(device)

    # --- load model per GPU ---

    model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)

    model.eval()

    # --- centerbias (must be per process) ---

    image = face()

    centerbias_template = np.load('centerbias_mit1003.npy')

    centerbias = zoom(

        centerbias_template,

        (320 / centerbias_template.shape[0], 512 / centerbias_template.shape[1]),

        order=0,

        mode='nearest'

    )

    centerbias -= logsumexp(centerbias)

    centerbias_tensor = torch.tensor([centerbias], dtype=torch.float32).to(device)

    centerbias_tensor = transforms.Resize((320, 512))(centerbias_tensor)

    results = {}

    for id in indices:

        img, _, bbox_relative, category = dataset[id]

        tg_loc = bbox_cordinates(bbox_relative, 512, 320)

        history_x, history_y = fixation_initialize()

        img_tensor = img.unsqueeze(0).to(device)

        img_tensor = transforms.Resize((320, 512))(img_tensor)

        count, max_search = 0, 999

        coef = torch.ones((1, 320, 512), device=device)

        path = []

        while count < max_search:

            fixation_history_x = np.array(history_x)

            fixation_history_y = np.array(history_y)

            x_hist_tensor = torch.tensor(

                [fixation_history_x[model.included_fixations]],

                device=device,

                dtype=torch.float32

            )

            y_hist_tensor = torch.tensor(

                [fixation_history_y[model.included_fixations]],

                device=device,

                dtype=torch.float32

            )

            with torch.no_grad():

                log_density = model(img_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)

            path.append([history_x[-1], history_y[-1]])

            isTg, coordinates, coef = logsearchProcess(

                history_x[-1],

                history_y[-1],

                tg_loc,

                log_density.squeeze(0).cpu(),

                (320, 512),

                48,

                coef

            )

            count += 1

            if isTg:

                break

            history_x.append(coordinates[0])

            history_y.append(coordinates[1])

        results[id] = count

    result_dict[gpu_id] = results

def run_parallel(dataset):

    num_gpus = 3

    indices = np.array_split(np.arange(len(dataset)), num_gpus)

    manager = mp.Manager()

    result_dict = manager.dict()

    processes = []

    for gpu_id in range(num_gpus):

        p = mp.Process(

            target=worker,

            args=(gpu_id, dataset, indices[gpu_id], result_dict)

        )

        p.start()

        processes.append(p)

    for p in processes:

        p.join()

    # merge results

    final_results = {}

    for d in result_dict.values():

        final_results.update(d)

    return final_results



def logsearchProcess(x, y, tg_xy, attentionMap, image_size, size, coef):
    mask_size = size
    tg_x, tg_y, w, h = tg_xy
    tg_xmax, tg_ymax = tg_x + w, tg_y + h 

    attenNP = (attentionMap[0,:,:].detach() * coef[0,:,:].detach()).numpy()
    y_fix, x_fix = y, x

    x_max_s, x_min_s, y_max_s, y_min_s = min(x_fix+mask_size//2, image_size[1]-1), max(x_fix-mask_size//2, 0), min(y_fix+mask_size//2, image_size[0]-1), max(y_fix-mask_size//2, 0)

    if x_max_s < tg_x or x_min_s > tg_xmax or y_max_s < tg_y or y_min_s > tg_ymax:
        coef[0, y_min_s:y_max_s+1, x_min_s:x_max_s+1] = 1000
        attenNP = (attentionMap[0,:,:].detach() * coef[0,:,:].detach()).numpy()
        y_fix, x_fix = np.unravel_index(attenNP.argmax(), attenNP.shape)
        return False, [x_fix, y_fix], coef

    return True, [], coef

def fixation_initialize():
    k, x_range, y_range = 4, 50, 30
    x_init, y_init = 1680//2, 1050//2
    ratio_horizontal, ratio_vertical = 1680/512, 1050/320
    random_num_x, random_num_y = random.sample(list(range(x_range)), 4), random.sample(list(range(y_range)), 4)
    x, y = [], []
    for i in range(4):
        choice_x, choice_y = random.choice([0, 1]), random.choice([0, 1])
        x_cord = int((x_init+random_num_x[i])/ratio_horizontal) if choice_x == 0 else int((x_init-random_num_x[i])/ratio_horizontal)
        y_cord = int((y_init+random_num_y[i])/ratio_vertical) if choice_y == 0 else int((y_init-random_num_y[i])/ratio_vertical)
        x.append(x_cord)
        y.append(y_cord)

    return x, y


# In[6]:


DEVICE = 'cuda'
img_size = (320, 512)

# you can use DeepGazeI or DeepGazeIIE
model = deepgaze_pytorch.DeepGazeIII(pretrained=True)

# use multiple GPUs

# move model to GPU
model = model.to(DEVICE)




image = face()
centerbias_template = np.load('centerbias_mit1003.npy')
# rescale to match image size
centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
# renormalize log density
centerbias -= logsumexp(centerbias)
centerbias_tensor = torch.tensor([centerbias], dtype=torch.float32).to(DEVICE)
centerbias_tensor = transforms.Resize(img_size)(centerbias_tensor)


# In[ ]:


size = 48
deepgaze_CON_0_25, deepgaze_CON_25_50 = [], []
deepgaze_INCON_0_25, deepgaze_INCON_25_50 = [], []
scanpath, deepgaze_attention_map = {}, {}


deepgaze_res = []


# selected_imgs = bin_info['con_(0, 25]'].tolist() + bin_info['con_(25, 50]'].tolist() + bin_info['incon_(0, 25]'].tolist() + bin_info['incon_(25, 50]'].tolist()

deepgaze_res = run_parallel(dataset)


#     if id in bin_info['con_(0, 25]'].tolist():
#         deepgaze_CON_0_25.append(count)
#     elif id in bin_info['con_(25, 50]'].tolist():
#         deepgaze_CON_25_50.append(count)

#     elif id in bin_info['incon_(0, 25]'].tolist():
#         deepgaze_INCON_0_25.append(count)
#     elif id in bin_info['incon_(25, 50]'].tolist():
#         deepgaze_INCON_25_50.append(count)

    # print("search times_{}: ".format(id), count)

# deepgaze_CON_res = deepgaze_CON_0_25 + deepgaze_CON_25_50
# deepgaze_INCON_res = deepgaze_INCON_0_25 + deepgaze_INCON_25_50
# deepgaze_res = deepgaze_CON_res + deepgaze_INCON_res


# In[ ]:


np.mean(deepgaze_res)


# In[ ]:


# np.mean(deepgaze_res), np.mean(deepgaze_CON_res), np.mean(deepgaze_INCON_res)


# In[ ]:


def sampleIncon(incon_bin_result, con_bin_result, times):
    sample_times = times
    nums = len(con_bin_result)
    res = np.array([0.0] * 25)

    for id in range(sample_times):
        temp = sample(incon_bin_result, nums)
        temp_accu = [0] + model_performance(temp, len(temp))
        res += np.array(temp_accu[:25])

    return (res/sample_times).tolist()

def balanced_accu(res_con, res_incon):
    res = []
    for i in range(25):
        res.append((res_con[i]+res_incon[i])/2)

    return res


# In[ ]:


deepgaze_accu = model_performance(deepgaze_res, len(deepgaze_res))


# In[ ]:


# times = 100
# deepgaze_CON_accu = [0] + model_performance(deepgaze_CON_res, len(deepgaze_CON_res))
# deepgaze_INCON_accu = sampleIncon(deepgaze_INCON_res, deepgaze_CON_res, times)
# deepgaze_accu = balanced_accu(deepgaze_CON_accu, deepgaze_INCON_accu)
# deepgaze_accu[:11], deepgaze_CON_accu[:11], deepgaze_INCON_accu[:11]


# In[ ]:


deepgaze_SCEGRAM_res = {}
deepgaze_SCEGRAM_res['combined_accu'] = deepgaze_accu
# deepgaze_SCEGRAM_res['con_accu'] = deepgaze_CON_accu
# deepgaze_SCEGRAM_res['incon_accu'] = deepgaze_INCON_accu
# deepgaze_SCEGRAM_res['con_[0,25)'] = deepgaze_CON_0_25
# deepgaze_SCEGRAM_res['con_[25,50)'] = deepgaze_CON_25_50
# deepgaze_SCEGRAM_res['incon_[0,25)'] = deepgaze_INCON_0_25
# deepgaze_SCEGRAM_res['incon_[25,50)'] = deepgaze_INCON_25_50
# deepgaze_SCEGRAM_res['scanpath'] = scanpath
# deepgaze_SCEGRAM_res['attention_map'] = deepgaze_attention_map


# In[ ]:


with open("../results/ContextBreak/ContextBreak_deepgaze_res.pkl", "wb") as tf:
    pickle.dump(deepgaze_SCEGRAM_res, tf)


# In[ ]:




