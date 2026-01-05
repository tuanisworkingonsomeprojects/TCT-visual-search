import sys
import json
import yaml
import pathlib
import pickle
import random
import copy

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms, utils
from torchvision.transforms.functional import to_pil_image, to_tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

import os
import os.path as osp

import PIL
from PIL import Image, ImageDraw
from tqdm import tqdm, trange
import cv2 as cv
# from google.colab.patches import cv2_imshow

from ml_collections import ConfigDict
from .core.dataset import COCODataset, COCODatasetWithID, COCODatasetZeroTarget, COCODatasetRandom, COCODatasetMix
from .core.config import save_config
from .core.model import Model

# extract info of pic for further analysis
def pic_analysis(dataset, index):
    image, target_image, bbox_relative, label, annotation_id = dataset[index]
    width, height = dataset.image_size
    x_min, y_min, w, h = int(bbox_relative[0]*width), int(bbox_relative[1]*height), int(bbox_relative[2]*width), int(bbox_relative[3]*height)
    label_name = dataset.idx2label[label]

    return image, target_image, (x_min, y_min, w, h), label_name

# add red rectangle to target
def target_rec(image_tensor, cordinates):
    image = transforms.ToPILImage()(image_tensor)
    x_min, y_min, w, h = cordinates
    draw = ImageDraw.Draw(image)
    draw.rectangle([x_min, y_min,x_min+w, y_min+h], outline=(255,0,0)) 
    return image

def bbox_cordinates(bbox_relative, width, height):
    x_min, y_min, w, h = int(bbox_relative[0]*width), int(bbox_relative[1]*height), int(bbox_relative[2]*width), int(bbox_relative[3]*height)

    return (x_min, y_min, w, h)

def searchProcess(tg_xy, attentionMap, image_size, size):
    mask_size = size
    tg_x, tg_y, w, h = tg_xy
    tg_xmax, tg_ymax = tg_x + w, tg_y + h 
    attentionMap[0, int(image_size[0]//2), int(image_size[1]//2)] = 1000
    maxSearch = 999

    count = 0 # num of search times of model following human's cordinates
    while True:
        attenNP = attentionMap[0,:,:].detach().numpy()
        y_fix, x_fix = np.unravel_index(attenNP.argmax(), attenNP.shape)
        count += 1

        # max search times, if exceeded, break
        if count >= maxSearch:
            break

        x_max_s, x_min_s, y_max_s, y_min_s = min(x_fix+mask_size//2, image_size[1]-1), max(x_fix-mask_size//2, 0), min(y_fix+mask_size//2, image_size[0]-1), max(y_fix-mask_size//2, 0)

        if x_max_s < tg_x or x_min_s > tg_xmax or y_max_s < tg_y or y_min_s > tg_ymax:
            attentionMap[0, y_min_s:y_max_s+1, x_min_s:x_max_s+1] = 0
        else:
            break
    return count

def model_performance(search_list, image_num):
    search_counter = pd.value_counts(search_list)
    max_search_times = max(search_list)
    sum_found_model = 0
    accu_model_performance = [0 for _ in range(max_search_times + 1)]
    for i in range(1, max_search_times + 1):
        try:
            sum_found_model += search_counter[i]
            accu_model_performance[i] = sum_found_model / image_num
        except KeyError:
            accu_model_performance[i] = sum_found_model / image_num

    return accu_model_performance

def searchProcesswithPath(tg_xy, attentionMap, image_size, size):
    searchPath = []
    mask_size = size
    tg_x, tg_y, w, h = tg_xy
    tg_xmax, tg_ymax = tg_x + w, tg_y + h 
    attentionMap[0, int(image_size[0]//2), int(image_size[1]//2)] = 1000
    maxSearch = 999

    count = 0 # num of search times of model following human's cordinates
    while True:
        attenNP = attentionMap[0,:,:].detach().numpy()
        y_fix, x_fix = np.unravel_index(attenNP.argmax(), attenNP.shape)
        searchPath.append([x_fix, y_fix])
        count += 1

        # max search times, if exceeded, break
        if count >= maxSearch:
            break

        x_max_s, x_min_s, y_max_s, y_min_s = min(x_fix+mask_size//2+1, image_size[1]-1), max(x_fix-mask_size//2, 0), min(y_fix+mask_size//2+1, image_size[0]-1), max(y_fix-mask_size//2, 0)

        if x_max_s < tg_x or x_min_s > tg_xmax or y_max_s < tg_y or y_min_s > tg_ymax:
            attentionMap[0, y_min_s:y_max_s+1, x_min_s:x_max_s+1] = 0
        else:
            break
    return count, searchPath

def scanPathShow(paths, attention, img_orignal=None):
    img_blend = None
    if img_orignal is not None: 
        img_blend = to_pil_image(attention * transforms.ToTensor()(img_orignal))
    else:
        img_blend = to_pil_image(attention)
        
    img = cv.cvtColor(np.asarray(img_blend),cv.COLOR_RGB2BGR) 
    # draw circle and numbers
    cv.circle(img,(256,160),5,(0,255,255),2,1)
    count = 0
    for path in paths:
        count += 1
        cx, cy = path[0], path[1]
        cv.circle(img,(cx, cy),5,(0,255,255),2,1)
        cv.putText(img, str(count), (cx, cy - 15), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 1)

    # draw the arrows
    for i in range(len(paths)):
        if i == 0:
            cx, cy = paths[i][0], paths[i][1]
            cv.arrowedLine(img, (256,160), (cx, cy), (0,255,255), 2, 0, 0, 0.1)
        else:
            cx_start, cy_start = paths[i-1][0], paths[i-1][1]
            cx_end, cy_end = paths[i][0], paths[i][1]
            cv.arrowedLine(img,(cx_start, cy_start), (cx_end, cy_end), (0,255,255), 2, 0, 0, 0.07)

    cv2_imshow(img)
    
def loadModel(checkpoint_dir, config_dir):
    with open(config_dir) as f:
        cfg = ConfigDict(yaml.load(f, Loader=yaml.Loader))

    if not hasattr(cfg, "num_classes"): # infer number of classes
        with open(cfg.annotations) as f:
            NUM_CLASSES = len(json.load(f)["categories"])
        cfg.num_classes = NUM_CLASSES

    # load model
    checkpoint = torch.load(checkpoint_dir, map_location="cpu")
    model = Model.from_config(cfg)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.extended_output = True
    model.training = False

    return model  

    
def CRTTest(input_images, model, layer_num, img_threshold, searcharea_size, image_resize):
    num_pics = len(input_images)
    layer_num, img_threshold, size, image_size = layer_num, img_threshold, searcharea_size, image_resize
    res_larger2, res_smaller2, res_larger2_cnt, res_smaller2_cnt, CRTNet_res = [], [], 0, 0, []
    attention_map, scan_paths = {}, []

    # set eval mode
    model.eval() 
    with torch.no_grad():
        for id in trange(num_pics):
            # get attention map from crtnet model
            context_images, target_images, bbox, labels_cpu = input_images[id]
            label_name = input_images.idx2label[labels_cpu]
            predicted_class, attention_CRTNet = model(context_images.unsqueeze(0), target_images.unsqueeze(0), bbox.unsqueeze(0))

            attention_CRTNet = attention_CRTNet.detach().squeeze()
            attention_CRTNet = attention_CRTNet.reshape(attention_CRTNet.size(0), 7, 7)
            mask_CRTNET = attention_CRTNet[layer_num]/attention_CRTNet[layer_num].max() # normalize
            mask_CRTNET = transforms.Resize(image_size)(mask_CRTNET.unsqueeze(0))
            attention_map[id] = copy.deepcopy(mask_CRTNET)

            _, predicted_label = torch.max(predicted_class.detach().to("cpu"), 1)
            predicted_label = predicted_label.item()

            tg_loc = bbox_cordinates(bbox, image_size[1], image_size[0])
            img_ratio = tg_loc[-1]*tg_loc[-2]/(image_size[1] * image_size[0])

            CRTNet_num, path = searchProcesswithPath(tg_loc, mask_CRTNET, image_size, size)
            CRTNet_res.append(CRTNet_num) 
            scan_paths.append(path) 

            if img_ratio > img_threshold:
                res_larger2.append(id)
                res_larger2_cnt += predicted_label==labels_cpu
            else:
                res_smaller2.append(id)
                res_smaller2_cnt += predicted_label==labels_cpu

            print(f' CRTNet_{str(id)}: {str(CRTNet_num)}', end = '\t')
            print(label_name + ": " + str(predicted_label==labels_cpu))

    overall_accu = (res_larger2_cnt+res_smaller2_cnt)/len(input_images)
    l2_accu, s2_accu = res_larger2_cnt/len(res_larger2), res_smaller2_cnt/len(res_smaller2)
    
    return [overall_accu, l2_accu, s2_accu], CRTNet_res, attention_map, scan_paths

def scanPathDraw(paths, attention, img_orignal=None):
    img_blend = None
    if img_orignal is not None: 
        img_blend = to_pil_image(attention * transforms.ToTensor()(img_orignal))
    else:
        img_blend = to_pil_image(attention)
        
    img = cv.cvtColor(np.asarray(img_blend),cv.COLOR_RGB2BGR) 
    # draw circle and numbers
    cv.circle(img,(256,160),5,(0,255,255),2,1)
    count = 0
    for path in paths:
        count += 1
        cx, cy = path[0], path[1]
        cv.circle(img,(cx, cy),5,(0,255,255),2,1)
        cv.putText(img, str(count), (cx, cy - 15), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 1)

    # draw the arrows
    for i in range(len(paths)):
        if i == 0:
            cx, cy = paths[i][0], paths[i][1]
            cv.arrowedLine(img, (256,160), (cx, cy), (0,255,255), 2, 0, 0, 0.1)
        else:
            cx_start, cy_start = paths[i-1][0], paths[i-1][1]
            cx_end, cy_end = paths[i][0], paths[i][1]
            cv.arrowedLine(img,(cx_start, cy_start), (cx_end, cy_end), (0,255,255), 2, 0, 0, 0.07)

    res = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB))

    return res

def image_grid(imgs, rows, cols, width, color):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*(w+len(imgs)*width), rows*(h)))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        w = w + width
        # w_temp = w-(i-1)*width
        img_new = Image.new('RGB', (w, h), color)
        grid.paste(img_new, box=(i%cols*w, i//cols*h))
        grid.paste(img, box=(i%cols*w, i//cols*h))
    
    return grid

def CRTAttention(input_images, model, layer_num, searcharea_size, image_resize, originalsize=(320, 512)):
    num_pics, layer_num, size, image_size = len(input_images), layer_num, searcharea_size, image_resize
    attention_map, random_file_pairs, random_idx_pairs = {}, {}, {}

    # set eval mode
    model.eval() 
    with torch.no_grad():
        for id in trange(num_pics):
            # get attention map from crtnet model
            context_images, target_images, bbox, labels_cpu, file_pair, idx_pair = input_images[id]
            label_name = input_images.idx2label[labels_cpu]
            _, attention_CRTNet = model(context_images.unsqueeze(0), target_images.unsqueeze(0), bbox.unsqueeze(0))

            attention_CRTNet = attention_CRTNet.detach().squeeze()
            attention_CRTNet = attention_CRTNet.reshape(attention_CRTNet.size(0), 7, 7)
            mask_CRTNET = attention_CRTNet[layer_num]
            mask_CRTNET = transforms.Resize(image_size)(mask_CRTNET.unsqueeze(0))
            
            file_name = input_images.id2file[input_images.annotations[id]['image_id']]
            if file_name.split('/')[-1][:len(label_name)] == label_name:
                dict_name = '_'.join([label_name, file_name.split('/')[-1][len(label_name):]])
            else:
                dict_name = '_'.join([label_name, file_name.split('/')[-1]])
            
            attention_map[dict_name] = copy.deepcopy(mask_CRTNET)
            random_file_pairs[file_pair[0]] = file_pair[1]
            random_idx_pairs[idx_pair[0]] = idx_pair[1]

            tg_loc = bbox_cordinates(bbox, originalsize[1], originalsize[0])
            mask_CRTNET = mask_CRTNET/mask_CRTNET.max()
            mask_CRTNET = transforms.Resize(originalsize)(mask_CRTNET)

            CRTNet_num, _ = searchProcesswithPath(tg_loc, mask_CRTNET, originalsize, size)

            print(f' CRTNet_{str(id)}: {str(CRTNet_num)}', end = '\n')
    
    return attention_map, random_file_pairs, random_idx_pairs 
    
def CRTAttentionFixedPairs(input_images, model, layer_num, searcharea_size, image_resize, random_idx_pairs, originalsize=(320, 512)):
    num_pics, layer_num, size, image_size = len(input_images), layer_num, searcharea_size, image_resize
    attention_map_original, attention_map_2032, CRTres, scan_path = {}, {}, [], []

    # set eval mode
    model.eval() 
    with torch.no_grad():
        for id in trange(num_pics):
            # get attention map from crtnet model
            context_images, _, bbox, labels_cpu = input_images[id]
            _, target_images, _, _ = input_images[random_idx_pairs[id]]
            label_name = input_images.idx2label[labels_cpu]
            
            # get the filename
            file_name = input_images.id2file[input_images.annotations[id]['image_id']]
            if file_name.split('/')[-1][:len(label_name)] == label_name:
                dict_name = '_'.join([label_name, file_name.split('/')[-1][len(label_name):]])
            else:
                dict_name = '_'.join([label_name, file_name.split('/')[-1]])
            
            # calculate attention map
            _, attention_CRTNet = model(context_images.unsqueeze(0), target_images.unsqueeze(0), bbox.unsqueeze(0))
            attention_CRTNet = attention_CRTNet.detach().squeeze()
            attention_CRTNet = attention_CRTNet.reshape(attention_CRTNet.size(0), 7, 7)
            mask_CRTNET = attention_CRTNet[layer_num]
            
            # orginal attention map
            attention_map_original[dict_name] = mask_CRTNET.unsqueeze(0)
            
            # 2032 attention map
            mask_CRTNET_2032 = copy.deepcopy(transforms.Resize(image_size)(mask_CRTNET.unsqueeze(0)))
            attention_map_2032[dict_name] = mask_CRTNET_2032

            tg_loc = bbox_cordinates(bbox, originalsize[1], originalsize[0])
            mask_CRTNET = mask_CRTNET/mask_CRTNET.max()
            mask_CRTNET = transforms.Resize(originalsize)(mask_CRTNET.unsqueeze(0))

            CRTNet_num, path = searchProcesswithPath(tg_loc, mask_CRTNET, originalsize, size)
            CRTres.append(CRTNet_num)
            scan_path.append(path)
            
            print(f' CRTNet_{str(id)}: {str(CRTNet_num)}', end = '\n')
    
    return attention_map_original, attention_map_2032, CRTres, scan_path 