import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.helpers import DropPath, drop_path
from models.caogao import slow_attn_diff, slow_attn_diff2, slow_attn_test
from models.dididi import slow_attn_cal_dis
# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']

threshold_all = 0.95
threshold_shallow = 0.95
threshold_deep = 0.90
# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None

def slow_attn_after_mask(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    # root_path = '/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/temp/'
    # os.makedirs(root_path, exist_ok=True)
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: attn.add_(attn_mask)
    
    level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    for index, this_square in enumerate(token_scale):
        if query.shape[2] == this_square:
            layer = index
            break
    
    if layer in range(0, 3):
        out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
    if layer in range(3,10):
        attention_map = torch.load(f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/attention16/attentionmap{layer}.pth')
        edge = token_scale[layer]
        left_edge2 = tokenall_scale[layer-2]
        right_edge2 = tokenall_scale[layer-1]
        left_edge3 = tokenall_scale[layer-3]
        right_edge3 = tokenall_scale[layer-2]
        left_edge4 = tokenall_scale[layer-4]
        right_edge4 = tokenall_scale[layer-3]
        print(f'last mask is {attention_map[0, 0, :, -edge:].shape}')

        '''一个mask的情况:layer=9'''
        if layer == 9:
            print(f'##################    BLOCK {block_num}    ######################')
            print(f'##################    SCALE {layer}    ######################')
            # 预先选定的模板窗口大小
            if query.shape[1] == 16:
                win_sizes1 = [16, 17, 18, 20, 30, 34, 37, 45, 47, 48, 50, 64] # 修改模板窗口大小
            elif query.shape[1] == 30:
                win_sizes1 = [16, 17, 32, 33, 34, 37, 48, 49, 50, 52, 54, 60, 64] # 修改模板窗口大小
            y1 = attention_map[:, :, :, -edge:].cpu() # attentionmap(16,16,169,424),y1(16,16,169,169)
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            
            for i in range(y_mean1.shape[0]):
                print(f'HEAD {i}')
                # 倒数第一个mask
                mask_tru1 = 0
                for win_size in win_sizes1:
                    mask1 = torch.abs(x_cor1 - y_cor1) <= win_size # ([16, 16]),后续诸如([169, 169])
                    attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                    if attn_fraction > threshold_deep:
                        new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                        print(f'HEAD {i} MASK 1 = {win_size}')
                        mask_tru1 += 1
                        break
                if mask_tru1 == 0:
                    new_attn1 = y1[:,i,:,:]
                    print(f'NO USE ANY MASK 1 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1
            print(F'Attn_aftermask.shape={attn_aftermask.shape}')
            out = attn_aftermask @ value

        '''两个mask的情况:layer=3'''
        if layer == 3:
            print(f'##################    BLOCK {block_num}    ######################')
            print(f'##################    SCALE {layer}    ######################')
            # 预先选定的模板窗口大小
            if query.shape[1] == 16:
                win_sizes1 = [5, 6, 7, 8, 9, 10, 11, 12, 15] # 修改模板窗口大小
                win_sizes2 = [4, 5, 6, 7, 10]
            elif query.shape[1] == 30:
                win_sizes1 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 20] # 修改模板窗口大小
                win_sizes2 = [5, 6, 7, 8, 15]
            y1 = attention_map[:, :, :, -edge:].cpu() # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            y_mean2 = y2.mean(dim=0)
            x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            x_cor2, y_cor2 = torch.meshgrid(torch.arange(y_mean2.shape[1]), torch.arange(y_mean2.shape[2]))
            
            for i in range(y_mean1.shape[0]):
                print(f'HEAD {i}')
                # 倒数第一个mask
                mask_tru1 = 0
                for win_size in win_sizes1:
                    mask1 = torch.abs(x_cor1 - y_cor1) <= win_size # ([16, 16]),后续诸如([169, 169])
                    attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        # new_attn1 = mask1 * y1
                        new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                        print(f'HEAD {i} MASK 1 = {win_size}')
                        mask_tru1 += 1
                        break
                    # else:
                    #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                if mask_tru1 == 0:
                    new_attn1 = y1[:,i,:,:]
                    print(f'NO USE ANY MASK 1 IN BLOCK {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1
                # 倒数第二个mask
                mask_tru2 = 0
                for win_size in win_sizes2:
                    mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size
                    attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        # new_attn1 = mask1 * y1
                        new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                        print(f'HEAD {i} MASK 2 = {win_size}')
                        mask_tru2 += 1
                        break
                if mask_tru2 == 0:
                    new_attn2 = y2[:,i,:,:]
                    print(f'NO USE ANY MASK 2 IN BLOCK {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2
            print(F'Attn_aftermask.shape={attn_aftermask.shape}')
            out = attn_aftermask @ value
        
        '''三个mask的情况:layer=4和layer=5'''
        if layer == 4 or layer == 5:
            print(f'##################    BLOCK {block_num}    ######################')
            print(f'##################    SCALE {layer}    ######################')
            if layer == 4:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 25]  # 修改模板窗口大小
                    win_sizes2 = [6, 7, 8, 9, 10, 11, 13, 15]
                    win_sizes3 = [4, 5, 6, 7, 10]
                elif query.shape[1] == 30:
                    win_sizes1 = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25] # 修改模板窗口大小
                    win_sizes2 = [6, 7, 8, 9, 10, 11, 12, 13, 15, 20]
                    win_sizes3 = [5, 6, 7, 8, 10]
            elif layer == 5:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [6, 7, 11, 12, 13, 18, 20, 25, 28, 30]  
                    win_sizes2 = [6, 7, 8, 9, 10, 13, 15, 17, 20, 25]
                    win_sizes3 = [5, 6, 7, 8, 10, 12, 13]
                elif query.shape[1] == 30:
                    win_sizes1 = [6, 8, 10, 12, 13, 16, 18, 22, 24, 26, 28, 30, 35, 45] # 修改模板窗口大小
                    win_sizes2 = [6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 25]
                    win_sizes3 = [6, 7, 8, 9, 10, 11, 12, 13, 15, 20]
            y1 = attention_map[:, :, :, -edge:].cpu() # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            y_mean2 = y2.mean(dim=0)
            y_mean3 = y3.mean(dim=0)
            x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            x_cor2, y_cor2 = torch.meshgrid(torch.arange(y_mean2.shape[1]), torch.arange(y_mean2.shape[2]))
            x_cor3, y_cor3 = torch.meshgrid(torch.arange(y_mean3.shape[1]), torch.arange(y_mean3.shape[2]))
            for i in range(y_mean1.shape[0]):
                print(f'HEAD {i}')
                # 倒数第一个mask
                mask_tru1 = 0
                for win_size in win_sizes1:
                    mask1 = torch.abs(x_cor1 - y_cor1) <= win_size # ([16, 16]),后续诸如([169, 169])
                    attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        # new_attn1 = mask1 * y1
                        new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                        print(f'HEAD {i} MASK 1 = {win_size}')
                        mask_tru1 += 1
                        break
                    # else:
                    #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                if mask_tru1 == 0:
                    new_attn1 = y1[:,i,:,:]
                    print(f'NO USE ANY MASK 1 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1
                # 倒数第二个mask
                mask_tru2 = 0
                for win_size in win_sizes2:
                    mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size
                    attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        # new_attn1 = mask1 * y1
                        new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                        print(f'HEAD {i} MASK 2 = {win_size}')
                        mask_tru2 += 1
                        break
                    # else:
                    #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                if mask_tru2 == 0:
                    new_attn2 = y2[:,i,:,:]
                    print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2
                # 倒数第三个mask
                mask_tru3 = 0
                for win_size in win_sizes3:
                    mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size
                    attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                        print(f'HEAD {i} MASK 3 = {win_size}')
                        mask_tru3 += 1
                        break
                if mask_tru3 == 0:
                    new_attn3 = y3
                    print(f'NO USE ANY MASK 3 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3
            print(F'Attn_aftermask.shape={attn_aftermask.shape}')
            out = attn_aftermask @ value

        '''四个mask的情况:layer=6和layer=7和layer=8'''
        if layer == 6 or layer == 7 or layer == 8:
            print(f'##################    BLOCK {block_num}    ######################')
            print(f'##################    SCALE {layer}    ######################')
            if layer == 6:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [8, 9, 11, 16, 18, 26, 30, 43, 51, 64]  # 修改模板窗口大小
                    win_sizes2 = [8, 9, 10, 11, 12, 13, 17, 24, 30, 35]
                    win_sizes3 = [6, 7, 8, 9, 10, 15, 20, 25]
                    win_sizes4 = [6, 7, 9, 11, 13, 20]
                elif query.shape[1] == 30:
                    win_sizes1 = [8, 9, 10, 13, 15, 16, 17, 20, 23, 24, 26, 40, 41, 42, 43, 44, 46, 48, 50, 64]  # 修改模板窗口大小
                    win_sizes2 = [5, 9, 10, 11, 13, 14, 16, 17, 23, 24, 25, 26, 27, 29, 30, 35]
                    win_sizes3 = [7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 25]
                    win_sizes4 = [5, 7, 9, 11, 12, 13, 20]
            elif layer == 7:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [11, 20, 21, 22, 25, 30, 50, 64]
                    win_sizes2 = [9, 10, 13, 18, 20, 31, 46, 51, 64]
                    win_sizes3 = [7, 9, 10, 12, 20, 25, 30, 40]
                    win_sizes4 = [7, 8, 9, 10, 11, 15, 18, 20, 30]
                elif query.shape[1] == 30:
                    win_sizes1 = [11, 20, 22, 25, 28, 30, 32, 35, 37, 40, 51, 56, 60, 64]  # 修改模板窗口大小
                    win_sizes2 = [8, 10, 13, 15, 16, 18, 20, 22, 24, 25, 30, 32, 38, 40, 42, 43, 44, 46, 48, 49, 50, 51, 64]
                    win_sizes3 = [6, 9, 10, 11, 13, 14, 17, 19, 21, 23, 25, 27, 28, 29, 35]
                    win_sizes4 = [5, 6, 8, 11, 12, 14, 15, 17, 19, 20, 25]
            elif layer == 8:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [14, 15, 28, 35, 39, 42, 50, 57, 64]
                    win_sizes2 = [11, 12, 20, 23, 24, 30, 35, 45, 64]
                    win_sizes3 = [9, 15, 17, 20, 32, 48, 51, 64]
                    win_sizes4 = [6, 7, 9, 11, 13, 15, 19, 26, 30, 40]
                elif query.shape[1] == 30:
                    win_sizes1 = [16, 19, 24, 27, 29, 31, 37, 39, 40, 41, 50, 52, 54, 58, 63]  # 修改模板窗口大小
                    win_sizes2 = [13, 19, 21, 23, 25, 27, 29, 31, 32, 33, 35, 36, 40, 42, 44, 54, 56, 57, 59, 60, 64]
                    win_sizes3 = [8, 10, 13, 15, 17, 19, 21, 22, 23, 24, 26, 28, 31, 35, 38, 40, 42, 44, 46, 48, 50, 51, 60]
                    win_sizes4 = [6, 8, 10, 11, 13, 14, 16, 19, 21, 23, 25, 26, 27, 28, 29, 30, 35]
            y1 = attention_map[:, :, :, -edge:].cpu() # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            y4 = attention_map[:, :, :, left_edge4:right_edge4].cpu()
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            y_mean2 = y2.mean(dim=0)
            y_mean3 = y3.mean(dim=0)
            y_mean4 = y4.mean(dim=0)
            x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            x_cor2, y_cor2 = torch.meshgrid(torch.arange(y_mean2.shape[1]), torch.arange(y_mean2.shape[2]))
            x_cor3, y_cor3 = torch.meshgrid(torch.arange(y_mean3.shape[1]), torch.arange(y_mean3.shape[2]))
            x_cor4, y_cor4 = torch.meshgrid(torch.arange(y_mean4.shape[1]), torch.arange(y_mean4.shape[2]))
            for i in range(y_mean1.shape[0]):
                print(f'HEAD {i}')
                # 倒数第一个mask
                mask_tru1 = 0
                for win_size in win_sizes1:
                    mask1 = torch.abs(x_cor1 - y_cor1) <= win_size # ([16, 16]),后续诸如([169, 169])
                    attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                    if attn_fraction > threshold_deep:
                        # new_attn1 = mask1 * y1
                        new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                        print(f'HEAD {i} MASK 1 = {win_size}')
                        mask_tru1 += 1
                        break
                    # else:
                    #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                if mask_tru1 == 0:
                    new_attn1 = y1[:,i,:,:]
                    print(f'NO USE ANY MASK 1 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1
                # 倒数第二个mask
                mask_tru2 = 0
                for win_size in win_sizes2:
                    mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size
                    attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        # new_attn1 = mask1 * y1
                        new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                        print(f'HEAD {i} MASK 2 = {win_size}')
                        mask_tru2 += 1
                        break
                    # else:
                    #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                if mask_tru2 == 0:
                    new_attn2 = y2[:,i,:,:]
                    print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2
                # 倒数第三个mask
                mask_tru3 = 0
                for win_size in win_sizes3:
                    mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size
                    attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                        print(f'HEAD {i} MASK 3 = {win_size}')
                        mask_tru3 += 1
                        break
                if mask_tru3 == 0:
                    new_attn3 = y3
                    print(f'NO USE ANY MASK 3 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3
                # 倒数第四个mask
                mask_tru4 = 0
                for win_size in win_sizes4:
                    mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) <= win_size
                    attn_fraction = (mask4.unsqueeze(0) * y4[:,i,:,:]).sum() / y4[:,i,:,:].sum()
                    if attn_fraction > threshold_all:
                        new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:] #
                        print(f'HEAD {i} MASK 4 = {win_size}')
                        mask_tru4 += 1
                        break
                if mask_tru4 == 0:
                    new_attn4 = y4
                    print(f'NO USE ANY MASK 4 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4
            print(F'Attn_aftermask.shape={attn_aftermask.shape}')
            out = attn_aftermask @ value

    return out



