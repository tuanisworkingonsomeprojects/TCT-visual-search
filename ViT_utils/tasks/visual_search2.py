import numpy as np
import copy
import cv2
import math
from itertools import product
from PIL import Image
import os
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset, RandomSampler, SequentialSampler
from scipy.special import softmax

from models.modeling_htm1 import VisionTransformer, CONFIGS

PRE_DIR_PATH = '../..'

def prep_data(target_idx, search_idxs, data_type, target_h=128, target_w=128, search_h=128*3, search_w=128*3,  patch_aligned=True):
    if patch_aligned:
        idx_mask = np.zeros([search_h, search_w])
        
        if data_type == 'cifar10':
            target_image = cv2.resize(cifar_images[target_idx], (target_w, target_h))
            search_image = np.zeros([search_h, search_w, 3])
            for i, (idx, (row, col)) in enumerate(zip(search_idxs, product(np.arange(3), np.arange(3)))):
                im = cv2.resize(cifar_images[idx], (target_w, target_h))
                search_image[row*target_h:(row+1)*target_h, col*target_w:(col+1)*target_w, :] = im
                idx_mask[row*target_h:(row+1)*target_h, col*target_w:(col+1)*target_w] = int(i+1)
        elif data_type == 'object':
            target_image = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/array/target/target_{}.jpg'.format(target_idx)))), (target_w, target_h))
            search_image = np.zeros([search_h, search_w])
            for i, (idx, (row, col)) in enumerate(zip(search_idxs, product(np.arange(3), np.arange(3)))):
                im = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/array/target/target_{}.jpg'.format(idx)))), (target_w, target_h))
                search_image[row*target_h:(row+1)*target_h, col*target_w:(col+1)*target_w] = im
                idx_mask[row*target_h:(row+1)*target_h, col*target_w:(col+1)*target_w] = int(i+1)
        search_image = np.uint8(search_image)
        solution = np.where(np.array(search_idxs) == target_idx)[0] + 1
    else:
        if data_type == 'object':
            idx_mask = cv2.resize(np.asarray(sio.loadmat(os.path.join(PRE_DIR_PATH, 'waldo_datasets/EvalPlotSupportingFunc/eval/saliencyMask/maskind.mat'))['maskind']), (search_w, search_h))
            solution = sio.loadmat(os.path.join(PRE_DIR_PATH, 'waldo_datasets/array/psy/gt.mat'))['gt'][0][:300][target_idx-1]
            target_image = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/{}/target/target_{}.jpg'.format('array', target_idx)))), (target_w, target_h))
            search_image = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/{}/stimuli/array_{}.jpg'.format('array', search_idxs)))), (search_w, search_h))
        elif data_type == 'naturaldesign':
            idx_mask = None
            solution = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/naturaldesign/gt/gt{}.jpg'.format(target_idx))))[:, :, 0], (search_w, search_h))
            target_image = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/{}/target/t{}.jpg'.format('naturaldesign', str(target_idx).zfill(3))))), (target_w, target_h))
            search_image = cv2.resize(np.asarray(Image.open(os.path.join(PRE_DIR_PATH, 'waldo_datasets/{}/stimuli/img{}.jpg'.format('naturaldesign', str(target_idx).zfill(3))))), (search_w, search_h))
    return target_image, search_image, idx_mask, solution

def load_model(state_dict=os.path.join(PRE_DIR_PATH, 'vit-visual-search/vitpytorch_cifar100_finetune_no_posembed.pt'), posembed=False, device='cpu'):
    config = CONFIGS["ViT-B_16"]
    vit = VisionTransformer(config, num_classes=100, zero_head=True, img_size=224, vis=True, posembed=posembed)
    if 'pt' in state_dict:
        
        
        
        
        # ______________ CUDA CODE _______________
        # vit.load_state_dict(torch.load(state_dict), strict=False)
        # ______________ CUDA CODE _______________
        
        
        # _______________ CPU CODE ________________
        # vit.load_state_dict(torch.load(state_dict, map_location = torch.device('cpu')), strict=False)
        # _______________ CPU CODE ________________
    
        vit.load_state_dict(torch.load(state_dict, map_location = torch.device(device)), strict=False)
        
        
        
        
        
        
        
    elif 'npz' in state_dict:
        vit.load_from(np.load(state_dict))
        
    # _________ CUDA CODE________
    # vit.cuda()
    # _________ CUDA CODE________
    
    vit.eval()
    return vit

def get_att_map(vit, s_im, crt_mod_method=None, crt_global_mod_layers=[], crt_global_mod=None, crt_local_mod=None, 
                t_im=None, t_embed=None, mod_last_n=12, mod=None, attn_type='attention', mod_type='q', 
                processed=False, device='cuda'): 
    if not processed:
        # prepare search image
        if len(s_im.shape) == 2:
            # pass target and search images through vit independently 
            s_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]), transforms.Lambda(lambda x: x.repeat(3,1,1))])
            t_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]), transforms.Lambda(lambda x: x.repeat(3,1,1))])
        elif len(s_im.shape) == 3:
            s_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            t_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        s_tensor = s_transform(s_im)[None].to(device)
        if t_im is not None:
            t_tensor = t_transform(t_im)[None].to(device)
    else:
        s_tensor = s_im[None].to(device)
        if t_im is not None:
            t_tensor = t_im[None].to(device)

    # get search image attention 
    if mod is None: # no target-based modulation 
        _, s_attentions, _, _, _, _, _, _, _ = vit([s_tensor], crt_mod_method=crt_mod_method, crt_global_mod_layers=crt_global_mod_layers, crt_global_mod=crt_global_mod, crt_local_mod=crt_local_mod, device=device)
        return torch.stack(s_attentions).cpu().detach().squeeze().numpy()

    elif mod == 'concatenate': # inherent target modulation
        if t_im is not None:
            _, mod_attentions, _, _, _, _, _, _, _ = vit([t_tensor, s_tensor], device=device)
            return torch.stack(mod_attentions).cpu().detach().squeeze().numpy()
        else:
            raise NotImplementedError('Target image has to be given for concatenate modulation!')

    elif mod == 'relevance': # relevance-based target modulation 
        if t_im is not None:
            _, t_attentions, t_querys, t_keys, _, _, _, _, _ = vit([t_tensor], device=device)
        else:
            _, t_attentions, t_querys, t_keys, _, _, _, _, _ = vit([], embedding_output=t_embed, device=device)
        query_mod = torch.stack(t_querys).squeeze()
        key_mod = torch.stack(t_keys).squeeze()        

        if mod_type == 'q+k': # modulate with target query and key
            qmod, kmod = query_mod[-mod_last_n:], key_mod[-mod_last_n:]
        elif mod_type == 'q': # modulate with target query only
            qmod, kmod = query_mod[-mod_last_n:], None
        elif mod_type == 'k': # modulate with target key only
            qmod, kmod = None, key_mod[-mod_last_n:]
        else:
            raise NotImplementedError('Modulation type {} not implemented!'.format(mod_type))

        _, mod_s_attentions, _, _, _, _, _, relevances, _ = vit([s_tensor], crt_mod_method=crt_mod_method, crt_global_mod_layers=crt_global_mod_layers, crt_global_mod=crt_global_mod, crt_local_mod=crt_local_mod, query_mod=qmod, key_mod=kmod, device=device)
        if attn_type == 'attention':
            return torch.stack(mod_s_attentions).cpu().detach().squeeze().numpy()
        elif attn_type == 'relevance':
            return torch.stack(relevances[-mod_last_n:]).cpu().detach().squeeze().numpy()

def get_attn(vit, search_image, crt_mod_method=None, crt_global_mod_layers=[], crt_global_mod=None, crt_local_mod=None, t_im=None, target_embedding=None, mod_last_n=12, mod=None, attn_type='attention', mod_type='q', renormalization=False, processed=False, device='cuda'):
    attn = get_att_map(vit, search_image, crt_mod_method=crt_mod_method, crt_global_mod_layers=crt_global_mod_layers, 
                        crt_global_mod=crt_global_mod, crt_local_mod=crt_local_mod, 
                        t_im=t_im, t_embed=target_embedding, mod_last_n=mod_last_n, 
                        mod=mod, attn_type=attn_type, mod_type=mod_type, processed=processed, device=device)

    if processed:
        search_h, search_w = search_image.shape[-2], search_image.shape[-1]
    else:
        search_h, search_w = search_image.shape[0], search_image.shape[1]

    if mod == 'concatenate':
        n_target_patches = int(t_im.shape[0]/16) * int(t_im.shape[1]/16)
        start_patch = n_target_patches + 1
    else:
        start_patch = 1

    if renormalization:
        att_mat = torch.as_tensor(attn)

        # Average the attention weights across all heads.
        att_mat = torch.mean(att_mat, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)              

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        # grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, start_patch:].reshape(int(search_h/16), int(search_w/16)).detach().numpy()
        resized_attn = cv2.resize(mask / mask.max(), (search_w, search_h))
    else:
        if attn_type == 'attention':
            attn = attn.mean(1)[-1, 0, start_patch:].reshape(int(search_h/16), int(search_w/16))# average across attention heads, take final layer, take class token
        elif attn_type == 'relevance':
            if len(attn.shape) < 3: # only the last layer is modulated:
                attn = attn.mean(0)[start_patch:].reshape(int(search_h/16), int(search_w/16))# average across attention heads, take final layer
            else:
                attn = attn.mean(1)[-1, start_patch:].reshape(int(search_h/16), int(search_w/16))# average across attention heads, take final layer
        resized_attn = cv2.resize(attn, (search_w, search_h))
    return resized_attn

def intersects(fixation, inh_size, solution):
    x, y = fixation
    x1max, x1min, y1max, y1min = x + inh_size + 1, x - inh_size, y + inh_size + 1, y - inh_size
    x2max, x2min, y2max, y2min = np.where(solution > 0)[1].max(), np.where(solution > 0)[1].min(), np.where(solution > 0)[0].max(), np.where(solution > 0)[0].min()
    return not (x1max < x2min or x1min > x2max or y1max < y2min or y1min > y2max)
    
def search_task(resized_attn, solution, idx_mask=None, inh_size=50, fix_method='relaxed', data_type='naturaldesign'):
    running_sim = resized_attn.copy()
    guess_history, scan_path = [], []
    
    if data_type == 'object':
        resized_attn *= (idx_mask > 0) # remove all non-zero attention towards the background
        running_sim = resized_attn.copy()
        while (running_sim > 0).any():
            maxidx = np.argmax(running_sim.ravel())
            scan_path.append([maxidx % resized_attn.shape[-1], maxidx // resized_attn.shape[-1]])
            guess = idx_mask.ravel()[maxidx]
            guess_history.append(guess)
            if guess != solution:
                running_sim[idx_mask == guess] = 0  # inhibition of return
                continue
            else: break
        return guess_history, np.stack(scan_path)

    elif data_type == 'naturaldesign':
        xmax = solution.shape[-1]
        ymax = solution.shape[-2]
        while (running_sim > 0).any():
            maxidx = np.argmax(running_sim.ravel())
            x, y = maxidx % resized_attn.shape[-1], maxidx // resized_attn.shape[-1]
            scan_path.append([x, y])
            # original_inh_size = 200 # height and width of the inhibition of return box in the original natural image resolution
            # zoom_fold = 1024 / search_image.shape[0]  # zoom of the input search image compared to the original image (assuming same zoom in height and width)
            # inh_size = int(original_inh_size / zoom_fold / 2)

            if fix_method == 'rigorous':
            # More rigorous method: requires the fixation center to be within the gt box
                on_target = y <= np.where(solution > 0)[0].max() and y >= np.where(solution > 0)[0].min() and x <= np.where(solution > 0)[1].max() and x >= np.where(solution > 0)[1].min()
            elif fix_method == 'relaxed':
            # More relaxing method: requires overlap between the fixation center bounding box and the gt box
                on_target = intersects((x, y), inh_size, solution)
            
            if on_target:
                break
            else:
                running_sim[np.max([0, y-inh_size]):np.min([y+inh_size+1, ymax]),np.max([0, x-inh_size]):np.min([x+inh_size+1, xmax])] = 0 # inhibition of return centered at the fixation center (x,y)
                continue
                
    return np.stack(scan_path)

def get_cdf(paths, max_steps=20, data_type='natural', task_type='invariant'):
    if data_type == 'object':
        if task_type == 'identical':
            paths = paths[180:]
        elif task_type == 'invariant':
            paths = paths[:180]
    num_steps = np.array([len(p) for p in paths])
    cumsum = np.cumsum([(num_steps == n).sum() for n in np.arange(1, max_steps+1)])
    cdf = cumsum / len(num_steps)
    return cdf

def add_init_fix(attn, search_w, search_h, intensity=100):
    # # resize attn if needed 
    # if attn.shape[0] != search_h or attn.shape[1] != search_w:
    #     attn_copy = transforms.Resize((search_h, search_w))(torch.as_tensor(attn[None])).squeeze().numpy() 
    # else:
    attn_copy = attn.copy()
    attn_copy[int(search_h/2), int(search_w/2)] = intensity
    return attn_copy