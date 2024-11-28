import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
# from models.quantize_all_me import WeightPCQuantizer, WeightPGQuantizer, WeightPCQuantileQuantizer, WeightPGQuantileQuantizer
# from models.quantize_all_me import BaseQuantizer, ActDynamicQuantizer, ActPGDynamicQuantizer
# from models.quant_self_attention import QuantLinear
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
# from global_vars import MODE
# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']
import argparse
DEBUG=False
# QUANT=2


# 创建解析器
parser = argparse.ArgumentParser(description="Process some integers.")

# 添加参数
parser.add_argument('--cuda_devices', type=str, default="4,", help='Specify the CUDA devices to use, e.g., "0,1" for using devices 0 and 1')
parser.add_argument('--model_depth', type=int, choices=[16, 20, 24, 30], required=True, help='Specify the model depth')
parser.add_argument('--mode', type=int, choices=[0, 1, 2, -1], required=True, help='Specify the mode: 0 for original, 1 for masked, 2 for global cfg, -1 for design mask')
parser.add_argument('--pic_num', type=int, default=250, help='Specify the number of images to generate')
parser.add_argument('--seed', type=int, default=0, help='Set the seed of the model')
parser.add_argument('--quant', type=int, default=0, help='no quant model') # quant=1:w8a8 
parser.add_argument('--strict', type=str, default=" ", help='force w4a4') # try: others w8a8 
parser.add_argument('--threshold', type=float, default=0.95, help='only calculate attn in mask') # quant=1:w8a8 
# 解析参数
args = parser.parse_args()
# 设置参数
cuda_devices = args.cuda_devices
MODEL_DEPTH = args.model_depth
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
MODE = args.mode
PIC_NUM = args.pic_num
seed = args.seed
QUANT = args.quant
strict = args.strict
threshold = args.threshold

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
block_cal_token_num_list_1 = {} # 创建一个空的字典来存储每个 block 的计数值和 mask后的token vs原本toke计数
block_cal_token_num_list_2 = {}
block_cal_token_num_list_3 = {}
block_cal_token_num_list_4 = {}
window_wid_chosen_1 = {}
window_wid_chosen_2 = {}
window_wid_chosen_3 = {}
window_wid_chosen_4 = {}
mask_list = {}
# root_path = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/{MODEL_DEPTH}/threshold_{threshold}/'
root_path = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/beifen/'
# 指定要加载的设备
device = torch.device('cuda:0')
for mask_idx in range(30):
    print(f'Loading pth: {root_path}mask_{mask_idx}.pth')
    mask_list[mask_idx] = torch.load(root_path + f'mask_{mask_idx}.pth', map_location=device)

def mask_inference(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    # 确保输入在GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    # 原始
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    # 判断mode
    if MODE == 1: # mask
        pass
    elif MODE == 2: 
        extracted_part = attn[:8]
        attn[-8:] = extracted_part
    elif MODE == 0:
        print('ERROR! ORIGINAL SHOULD NOT USE THIS FUNCTION!')

    if attn_mask is not None: attn.add_(attn_mask)
    level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    for index, this_square in enumerate(token_scale):
        if query.shape[2] == this_square:
            layer = index
            break
    edge = token_scale[layer]
    left_edge2 = tokenall_scale[layer-2]
    right_edge2 = tokenall_scale[layer-1]
    left_edge3 = tokenall_scale[layer-3]
    right_edge3 = tokenall_scale[layer-2]
    left_edge4 = tokenall_scale[layer-4]
    right_edge4 = tokenall_scale[layer-3]
    left_edge5 = tokenall_scale[layer-5]
    right_edge5 = tokenall_scale[layer-4]
    left_edge6 = tokenall_scale[layer-6]
    right_edge6 = tokenall_scale[layer-5]
    left_edge7 = tokenall_scale[layer-7]
    right_edge7 = tokenall_scale[layer-6]
    attention_map = attn
    y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
    y2 = attention_map[:, :, :, left_edge2:right_edge2]
    y3 = attention_map[:, :, :, left_edge3:right_edge3]
    y4 = attention_map[:, :, :, left_edge4:right_edge4]
    y5 = attention_map[:, :, :, left_edge5:right_edge5]
    y6 = attention_map[:, :, :, left_edge6:right_edge6]
    y7 = attention_map[:, :, :, left_edge7:right_edge7]
    if layer in range(0, 3):
        out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
    if layer in range(3,10):
        # all_one = torch.ones_like(attention_map)
        '''一个mask的情况:layer=9'''
        if layer == 9:
            for i in range(attention_map.shape[1]): # which head
                # 倒数第一个mask
                win_size_1 = mask_list[23][block_num][i]
                win_size_2 = mask_list[24][block_num][i]
                win_size_3 = mask_list[25][block_num][i]
                win_size_4 = mask_list[26][block_num][i]
                win_size_5 = mask_list[27][block_num][i]
                win_size_6 = mask_list[28][block_num][i]
                win_size_7 = mask_list[29][block_num][i]

                attn_aftermask = attention_map

                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 倒数第六个mask              
                new_attn6 = win_size_6 * y6[:,i,:,:]
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                # 倒数第七个mask              
                new_attn7 = win_size_7 * y7[:,i,:,:]
                attn_aftermask[:, i, :, left_edge7:right_edge7] = new_attn7

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))

            out = attn_softmax @ value

        '''两个mask的情况:layer=3'''
        if layer == 3:
            for i in range(attention_map.shape[1]):
                # breakpoint()
                # 获取对应的 mask 
                win_size_1 = mask_list[0][block_num][i] ##################################################
                win_size_2 = mask_list[1][block_num][i] ##################################################

                attn_aftermask = attention_map
                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))

            out = attn_softmax @ value
        
        '''三个mask的情况:layer=4和layer=5'''
        if layer == 4 or layer == 5:
            y_mean1 = y1.mean(dim=0)
            for i in range(attention_map.shape[1]):
                # 获取对应的 window 宽度
                if layer == 4:
                    win_size_1 = mask_list[2][block_num][i] ##################################################
                    win_size_2 = mask_list[3][block_num][i] ##################################################
                    win_size_3 = mask_list[4][block_num][i] ##################################################
                    
                elif layer == 5:
                    win_size_1 = mask_list[5][block_num][i] ##################################################
                    win_size_2 = mask_list[6][block_num][i] ##################################################
                    win_size_3 = mask_list[7][block_num][i] ##################################################
                attn_aftermask = attention_map
                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))

            out = attn_softmax @ value

        '''四个mask'''
        if layer == 6:

            for i in range(attention_map.shape[1]):
                # 获取存好的mask
                win_size_1 = mask_list[8][block_num][i] ##################################################
                win_size_2 = mask_list[9][block_num][i] ##################################################
                win_size_3 = mask_list[10][block_num][i] ##################################################
                win_size_4 = mask_list[11][block_num][i] ##################################################

                attn_aftermask = attention_map
                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3
                
                # 倒数第四个mask
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))
            out = attn_softmax @ value

        '''五个mask'''
        if layer == 7:

            for i in range(attention_map.shape[1]): # which head

                win_size_1 = mask_list[12][block_num][i] ##################################################
                win_size_2 = mask_list[13][block_num][i]
                win_size_3 = mask_list[14][block_num][i]
                win_size_4 = mask_list[15][block_num][i]
                win_size_5 = mask_list[16][block_num][i]
            
                attn_aftermask = attention_map

                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))
            out = attn_softmax @ value
            
        '''六个mask'''
        if layer == 8:

            for i in range(attention_map.shape[1]): # which head

                win_size_1 = mask_list[17][block_num][i] ##################################################
                win_size_2 = mask_list[18][block_num][i]
                win_size_3 = mask_list[19][block_num][i]
                win_size_4 = mask_list[20][block_num][i]
                win_size_5 = mask_list[21][block_num][i]
                win_size_6 = mask_list[22][block_num][i]

                attn_aftermask = attention_map

                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 倒数第六个mask              
                new_attn6 = win_size_6 * y6[:,i,:,:]
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))
            
            out = attn_softmax @ value
    
    return out

# attn:int16, attention_map:int8, value:int8, out:int16
def quant_mask_inference(query, key, value, attn, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    # 确保输入在GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = value.to(device)

    # 判断mode
    if MODE == 1: # mask
        pass
    elif MODE == 2: 
        extracted_part = attn[:8] # int8
        attn[-8:] = extracted_part
    elif MODE == 0:
        print('ERROR! ORIGINAL SHOULD NOT USE THIS FUNCTION!')

    if attn_mask is not None: attn.add_(attn_mask)
    level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    for index, this_square in enumerate(token_scale):
        if query.shape[2] == this_square:
            layer = index
            break
    edge = token_scale[layer]
    left_edge2 = tokenall_scale[layer-2]
    right_edge2 = tokenall_scale[layer-1]
    left_edge3 = tokenall_scale[layer-3]
    right_edge3 = tokenall_scale[layer-2]
    left_edge4 = tokenall_scale[layer-4]
    right_edge4 = tokenall_scale[layer-3]
    left_edge5 = tokenall_scale[layer-5]
    right_edge5 = tokenall_scale[layer-4]
    left_edge6 = tokenall_scale[layer-6]
    right_edge6 = tokenall_scale[layer-5]
    left_edge7 = tokenall_scale[layer-7]
    right_edge7 = tokenall_scale[layer-6]
    attention_map = attn
    y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
    y2 = attention_map[:, :, :, left_edge2:right_edge2]
    y3 = attention_map[:, :, :, left_edge3:right_edge3]
    y4 = attention_map[:, :, :, left_edge4:right_edge4]
    y5 = attention_map[:, :, :, left_edge5:right_edge5]
    y6 = attention_map[:, :, :, left_edge6:right_edge6]
    y7 = attention_map[:, :, :, left_edge7:right_edge7]
    if layer in range(0, 3):
        out = (F.dropout(attention_map, p=dropout_p, inplace=True) if dropout_p > 0 else attention_map) @ value
    if layer in range(3,10):
        # all_one = torch.ones_like(attention_map)
        '''一个mask的情况:layer=9'''
        if layer == 9:
            for i in range(attention_map.shape[1]): # which head
                # 倒数第一个mask
                win_size_1 = mask_list[23][block_num][i]
                win_size_2 = mask_list[24][block_num][i]
                win_size_3 = mask_list[25][block_num][i]
                win_size_4 = mask_list[26][block_num][i]
                win_size_5 = mask_list[27][block_num][i]
                win_size_6 = mask_list[28][block_num][i]
                win_size_7 = mask_list[29][block_num][i]

                attn_aftermask = attention_map

                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 倒数第六个mask              
                new_attn6 = win_size_6 * y6[:,i,:,:]
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                # 倒数第七个mask              
                new_attn7 = win_size_7 * y7[:,i,:,:]
                attn_aftermask[:, i, :, left_edge7:right_edge7] = new_attn7

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask, p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask)

            out = attn_softmax @ value

        '''两个mask的情况:layer=3'''
        if layer == 3:
            for i in range(attention_map.shape[1]):
                # 获取对应的 mask 
                win_size_1 = mask_list[0][block_num][i] ##################################################
                win_size_2 = mask_list[1][block_num][i] ##################################################

                attn_aftermask = attention_map
                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask, p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask)

            out = attn_softmax @ value
        
        '''三个mask的情况:layer=4和layer=5'''
        if layer == 4 or layer == 5:
            y_mean1 = y1.mean(dim=0)
            for i in range(attention_map.shape[1]):
                # 获取对应的 window 宽度
                if layer == 4:
                    win_size_1 = mask_list[2][block_num][i] ##################################################
                    win_size_2 = mask_list[3][block_num][i] ##################################################
                    win_size_3 = mask_list[4][block_num][i] ##################################################
                    
                elif layer == 5:
                    win_size_1 = mask_list[5][block_num][i] ##################################################
                    win_size_2 = mask_list[6][block_num][i] ##################################################
                    win_size_3 = mask_list[7][block_num][i] ##################################################
                attn_aftermask = attention_map
                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask, p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask)

            out = attn_softmax @ value

        '''四个mask'''
        if layer == 6:

            for i in range(attention_map.shape[1]):
                # 获取存好的mask
                win_size_1 = mask_list[8][block_num][i] ##################################################
                win_size_2 = mask_list[9][block_num][i] ##################################################
                win_size_3 = mask_list[10][block_num][i] ##################################################
                win_size_4 = mask_list[11][block_num][i] ##################################################

                attn_aftermask = attention_map
                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3
                
                # 倒数第四个mask
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask, p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask)
            out = attn_softmax @ value

        '''五个mask'''
        if layer == 7:

            for i in range(attention_map.shape[1]): # which head

                win_size_1 = mask_list[12][block_num][i] ##################################################
                win_size_2 = mask_list[13][block_num][i]
                win_size_3 = mask_list[14][block_num][i]
                win_size_4 = mask_list[15][block_num][i]
                win_size_5 = mask_list[16][block_num][i]
            
                attn_aftermask = attention_map

                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask, p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask)
            out = attn_softmax @ value
            
        '''六个mask'''
        if layer == 8:

            for i in range(attention_map.shape[1]): # which head

                win_size_1 = mask_list[17][block_num][i] ##################################################
                win_size_2 = mask_list[18][block_num][i]
                win_size_3 = mask_list[19][block_num][i]
                win_size_4 = mask_list[20][block_num][i]
                win_size_5 = mask_list[21][block_num][i]
                win_size_6 = mask_list[22][block_num][i]

                attn_aftermask = attention_map

                # 倒数第一个mask
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 倒数第六个mask              
                new_attn6 = win_size_6 * y6[:,i,:,:]
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask, p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask)
            
            out = attn_softmax @ value
    
    return out