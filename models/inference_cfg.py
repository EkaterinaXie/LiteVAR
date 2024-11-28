import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from models.helpers import DropPath, drop_path
from models.caogao import slow_attn_diff, slow_attn_diff2, slow_attn_test
from models.dididi import slow_attn_cal_dis
# from global_vars import MODEL_DEPTH, MODE

# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']
os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
MODEL_DEPTH = 30
def initialize_with_params_cfg(cuda_devices, model_depth, mode, pic_num, seed):
    global CUDA_DEVICES, MODEL_DEPTH, MODE, PIC_NUM, SEED
    CUDA_DEVICES = cuda_devices
    MODEL_DEPTH = model_depth
    MODE = mode
    PIC_NUM = pic_num
    SEED = seed
    # 根据参数设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES


threshold_all = 0.95
threshold_shallow = 0.95
threshold_deep = 0.95
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
root_path = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/{MODEL_DEPTH}/'
# 指定要加载的设备
device = torch.device('cuda:0')
for mask_idx in range(30):
    print(f'Loading pth: mask_{mask_idx}.pth')
    mask_list[mask_idx] = torch.load(root_path + f'mask_{mask_idx}.pth', map_location=device)

def mask_inference_cfg_global(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    # 确保输入在GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    # 原始
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    # cfg 复用
    extracted_part = attn[:8]
    attn[-8:] = extracted_part

    if attn_mask is not None: attn.add_(attn_mask)
    # attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
    level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    for index, this_square in enumerate(token_scale):
        if query.shape[2] == this_square:
            layer = index
            break

    # # 打开文本文件进行读取
    # # with open(f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_txt/cal_tokens_aftermask_all_{model_depth}.txt', 'r') as file:
    # # with open(f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_txt/test_9_{model_depth}.txt', 'r') as file:
    # with open(f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_txt/try_all_3.txt', 'r') as file:
    #     content = file.read()
    # # 使用正则表达式匹配所有的window_list字典
    # pattern = r'Window we choose:\s*{([^}]+)}'
    # matches = re.findall(pattern, content)

    # # 保存匹配到的字典对象
    # window_lists = []
    # for match in matches:
    #     # 将匹配到的字符串转换为字典对象
    #     window_dict = eval('{' + match + '}')
    #     window_lists.append(window_dict)
    # '''window_lists[i]可以提取出第i个字典
    # 形式如：{0: [12, 13, 13, 11, 11, 13, 12, 13, 13, 12, 13, 12, 13, 13, 12, 12, 12, 13, 12, 12, 12, 12, 12, 13, 20, 13, 12, 10, 5, 13], 1: [5, 20, 10, 10, 12, 5, 13, 5, 13, 12, 13, 13, 12, 13, 13, 5, 5, 12, 10, 13, 12, 5, 11, 12, 8, 8, 5, 13, 5, 12], 2: [6, 8, 11, 5, 11, 9, 13, 12, 13, 7, 13, 8, 8, 9, 5, 8, 5, 12, 9, 20, 6, 6, 13, 7, 8, 12, 8, 5, 5, 12], 3: [12, 13, 12, 12, 5, 10, 5, 5, 12, 5, 12, 8, 13, 8, 5, 13, 13, 7, 13, 12, 12, 5, 12, 8, 12, 8, 12, 6, 5, 5], 4: [6, 12, 8, 7, 5, 12, 9, 6, 6, 10, 8, 5, 12, 12, 13, 13, 5, 12, 11, 10, 13, 6, 8, 5, 6, 13, 12, 12, 6, 8], 5: [12, 6, 12, 13, 8, 5, 12, 5, 11, 13, 13, 10, 12, 12, 11, 12, 12, 9, 5, 12, 6, 13, 7, 5, 10, 7, 13, 12, 13, 12], 6: [5, 8, 12, 9, 8, 12, 5, 12, 8, 13, 5, 5, 5, 5, 8, 12, 12, 12, 13, 6, 8, 13, 11, 8, 12, 12, 12, 13, 6, 8], 7: [13, 5, 13, 12, 9, 8, 12, 12, 11, 13, 9, 5, 8, 10, 5, 5, 5, 5, 11, 20, 12, 13, 11, 5, 9, 11, 9, 5, 5, 11], 8: [20, 10, 20, 11, 7, 12, 13, 12, 6, 12, 12, 8, 12, 10, 10, 12, 8, 11, 8, 13, 6, 12, 13, 5, 12, 12, 9, 10, 5, 10], 9: [13, 13, 8, 13, 7, 7, 7, 9, 12, 13, 11, 12, 12, 7, 12, 5, 13, 13, 13, 10, 12, 9, 12, 8, 12, 13, 12, 11, 5, 6], 10: [13, 9, 12, 12, 8, 8, 13, 13, 12, 9, 5, 8, 12, 9, 5, 13, 11, 12, 7, 11, 13, 8, 8, 5, 12, 6, 13, 10, 8, 13], 11: [13, 7, 7, 10, 11, 8, 12, 5, 5, 11, 13, 8, 11, 13, 5, 12, 11, 5, 8, 12, 12, 12, 13, 11, 13, 7, 11, 11, 12, 5], 12: [10, 12, 12, 7, 12, 12, 12, 12, 11, 12, 13, 9, 7, 13, 6, 12, 8, 13, 10, 12, 6, 11, 11, 12, 12, 12, 9, 9, 9, 13], 13: [12, 12, 5, 13, 11, 13, 12, 12, 12, 5, 6, 11, 8, 11, 9, 11, 12, 5, 10, 13, 12, 13, 12, 12, 7, 13, 12, 12, 11, 11], 14: [12, 6, 12, 11, 5, 13, 12, 10, 12, 6, 11, 12, 9, 13, 12, 5, 11, 12, 5, 13, 12, 8, 11, 13, 6, 11, 9, 10, 9, 12], 15: [11, 7, 9, 11, 13, 12, 13, 9, 8, 12, 8, 11, 12, 11, 6, 8, 12, 10, 5, 7, 12, 11, 10, 12, 11, 8, 12, 8, 7, 7], 16: [12, 11, 5, 7, 9, 10, 9, 5, 10, 12, 11, 5, 11, 7, 8, 12, 12, 12, 5, 12, 12, 12, 10, 11, 7, 7, 12, 12, 13, 8], 17: [5, 9, 10, 13, 5, 6, 5, 12, 10, 5, 10, 6, 5, 5, 11, 9, 6, 9, 8, 10, 5, 5, 12, 10, 5, 5, 5, 10, 5, 6], 18: [5, 11, 5, 12, 5, 10, 8, 7, 5, 8, 12, 6, 5, 5, 11, 8, 5, 8, 12, 10, 7, 9, 9, 11, 5, 11, 12, 12, 5, 5], 19: [8, 7, 6, 12, 5, 11, 13, 5, 5, 10, 5, 9, 7, 11, 7, 7, 12, 12, 10, 11, 5, 6, 8, 6, 5, 12, 9, 5, 8, 8], 20: [5, 12, 8, 6, 5, 9, 5, 8, 10, 12, 12, 12, 8, 9, 6, 8, 7, 11, 5, 10, 5, 10, 10, 5, 5, 13, 5, 10, 11, 10], 21: [5, 7, 5, 5, 5, 8, 11, 5, 6, 9, 6, 8, 5, 5, 5, 6, 11, 5, 8, 9, 9, 5, 13, 5, 8, 5, 7, 5, 9, 10], 22: [6, 7, 5, 5, 5, 5, 5, 5, 8, 7, 5, 11, 5, 8, 10, 5, 12, 5, 10, 5, 6, 10, 11, 8, 8, 10, 5, 8, 5, 5], 23: [5, 5, 8, 5, 11, 7, 5, 6, 8, 5, 12, 10, 8, 12, 5, 5, 7, 5, 5, 5, 5, 5, 5, 5, 5, 13, 10, 5, 5, 5], 24: [12, 20, 13, 13, 12, 13, 13, 11, 13, 13, 13, 12, 13, 12, 13, 13, 11, 12, 12, 12, 13, 13, 13, 13, 13, 12, 11, 12, 13, 13], 25: [8, 13, 8, 9, 13, 8, 13, 7, 5, 5, 8, 6, 11, 11, 7, 7, 5, 7, 12, 12, 11, 13, 13, 12, 9, 12, 11, 12, 8, 9], 26: [5, 5, 12, 11, 5, 5, 5, 5, 10, 5, 12, 5, 5, 5, 12, 5, 5, 5, 5, 5, 7, 5, 5, 12, 8, 5, 7, 10, 5, 9], 27: [5, 8, 5, 5, 5, 7, 5, 8, 6, 12, 5, 6, 12, 11, 6, 5, 9, 7, 5, 5, 5, 5, 7, 5, 7, 5, 12, 8, 5, 7], 28: [13, 5, 7, 7, 5, 12, 8, 11, 5, 7, 5, 10, 5, 6, 11, 5, 5, 13, 7, 11, 12, 5, 7, 6, 8, 5, 5, 7, 12, 5], 29: [12, 12, 7, 9, 7, 11, 9, 8, 11, 12, 12, 5, 8, 12, 11, 8, 8, 6, 5, 10, 8, 12, 9, 8, 12, 9, 7, 11, 12, 8]}
    # 0代表block的index,12,13……代表head 0-29 所选的mask窗口宽度
    # '''

    # torch.save(attention_map, root_path + f"attentionmap{ layer}.pth")
    if layer in range(0, 3):
        out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
    if layer in range(3,10):
        attention_map = attn
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
        all_one = torch.ones_like(attention_map)
        '''一个mask的情况:layer=9'''
        if layer == 9:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2]
            y3 = attention_map[:, :, :, left_edge3:right_edge3]
            y4 = attention_map[:, :, :, left_edge4:right_edge4]
            y5 = attention_map[:, :, :, left_edge5:right_edge5]
            y6 = attention_map[:, :, :, left_edge6:right_edge6]
            y7 = attention_map[:, :, :, left_edge7:right_edge7]
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            grid_list = []
            # for i in range(layer):
            #     x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
            #     grid_list.append((x_cor, y_cor))
            # x_cor1,y_cor1 = grid_list[0]
            # x_cor2,y_cor2 = grid_list[1]
            # x_cor3,y_cor3 = grid_list[2]
            # x_cor4,y_cor4 = grid_list[3]
            # x_cor5,y_cor5 = grid_list[4]
            # x_cor6,y_cor6 = grid_list[5]
            # x_cor7,y_cor7 = grid_list[6]
            for i in range(y_mean1.shape[0]): # which head
                # print(f'HEAD {i}')
                # 倒数第一个mask
                # 获取对应的 window 宽度
                # win_size_1 = window_lists[23][block_num][i] ##################################################
                # win_size_2 = window_lists[24][block_num][i]
                # win_size_3 = window_lists[25][block_num][i]
                # win_size_4 = window_lists[26][block_num][i]
                # win_size_5 = window_lists[27][block_num][i]
                # win_size_6 = window_lists[28][block_num][i]
                # win_size_7 = window_lists[29][block_num][i]

                win_size_1 = mask_list[23][block_num][i] ##################################################
                win_size_2 = mask_list[24][block_num][i]
                win_size_3 = mask_list[25][block_num][i]
                win_size_4 = mask_list[26][block_num][i]
                win_size_5 = mask_list[27][block_num][i]
                win_size_6 = mask_list[28][block_num][i]
                win_size_7 = mask_list[29][block_num][i]

                attn_aftermask = attention_map

                # 倒数第一个mask
                # mask1 = torch.abs(x_cor1 - y_cor1) <= win_size_1
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                # mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size_2
                # new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                # mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size_3
                # new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:]
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                # mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) <= win_size_4
                # new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:]
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                # mask5 = torch.abs(x_cor5*(right_edge5-left_edge5)/edge - y_cor5) <= win_size_5
                # new_attn5 = mask5.unsqueeze(0) * y5[:,i,:,:]
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 倒数第六个mask              
                # mask6 = torch.abs(x_cor6*(right_edge6-left_edge6)/edge - y_cor6) <= win_size_6
                # new_attn6 = mask6.unsqueeze(0) * y6[:,i,:,:]
                new_attn6 = win_size_6 * y6[:,i,:,:]
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                # 倒数第七个mask              
                # mask7 = torch.abs(x_cor7*(right_edge7-left_edge7)/edge - y_cor7) <= win_size_7
                # new_attn7 = mask7.unsqueeze(0) * y7[:,i,:,:]
                new_attn7 = win_size_7 * y7[:,i,:,:]
                attn_aftermask[:, i, :, left_edge7:right_edge7] = new_attn7

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))

            out = attn_softmax @ value

        '''两个mask的情况:layer=3'''
        if layer == 3:
            y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2]
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            y_mean2 = y2.mean(dim=0)

            # x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            # print("############")
            # print(x_cor1.device)
            # x_cor2, y_cor2 = torch.meshgrid(torch.arange(y_mean2.shape[1]), torch.arange(y_mean2.shape[2]))
            for i in range(y_mean1.shape[0]):
                # print(f'HEAD {i}')
                # 获取对应的 window 宽度
                # win_size_1 = window_lists[0][block_num][i] ##################################################
                # win_size_2 = window_lists[1][block_num][i] ##################################################

                win_size_1 = mask_list[0][block_num][i] ##################################################
                win_size_2 = mask_list[1][block_num][i] ##################################################

                attn_aftermask = attention_map
                # 倒数第一个mask
                # mask1 = torch.abs(x_cor1 - y_cor1) <= win_size_1
                # new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]

                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                # mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size_2
                # new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))

            out = attn_softmax @ value
        
        '''三个mask的情况:layer=4和layer=5'''
        if layer == 4 or layer == 5:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2]
            y3 = attention_map[:, :, :, left_edge3:right_edge3]
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            y_mean2 = y2.mean(dim=0)
            y_mean3 = y3.mean(dim=0)
            # x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            # x_cor2, y_cor2 = torch.meshgrid(torch.arange(y_mean2.shape[1]), torch.arange(y_mean2.shape[2]))
            # x_cor3, y_cor3 = torch.meshgrid(torch.arange(y_mean3.shape[1]), torch.arange(y_mean3.shape[2]))
            for i in range(y_mean1.shape[0]):
                # print(f'HEAD {i}')
                # 获取对应的 window 宽度
                if layer == 4:
                    # win_size_1 = window_lists[2][block_num][i] ##################################################
                    # win_size_2 = window_lists[3][block_num][i] ##################################################
                    # win_size_3 = window_lists[4][block_num][i] ##################################################

                    win_size_1 = mask_list[2][block_num][i] ##################################################
                    win_size_2 = mask_list[3][block_num][i] ##################################################
                    win_size_3 = mask_list[4][block_num][i] ##################################################
                    
                elif layer == 5:
                    # win_size_1 = window_lists[5][block_num][i] ##################################################
                    # win_size_2 = window_lists[6][block_num][i] ##################################################
                    # win_size_3 = window_lists[7][block_num][i] ##################################################

                    win_size_1 = mask_list[5][block_num][i] ##################################################
                    win_size_2 = mask_list[6][block_num][i] ##################################################
                    win_size_3 = mask_list[7][block_num][i] ##################################################
                attn_aftermask = attention_map
                # 倒数第一个mask
                # mask1 = torch.abs(x_cor1 - y_cor1) <= win_size_1
                # new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                # mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size_2
                # new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask
                # mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size_3
                # new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:]
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))

            out = attn_softmax @ value

        '''四个mask'''
        if layer == 6:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2]
            y3 = attention_map[:, :, :, left_edge3:right_edge3]
            y4 = attention_map[:, :, :, left_edge4:right_edge4]
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            grid_list = []
            # for i in range(layer):
            #     x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
            #     grid_list.append((x_cor, y_cor))
            # x_cor1,y_cor1 = grid_list[0]
            # x_cor2,y_cor2 = grid_list[1]
            # x_cor3,y_cor3 = grid_list[2]
            # x_cor4,y_cor4 = grid_list[3]
            
            for i in range(y_mean1.shape[0]):
                # win_size_1 = window_lists[8][block_num][i] ##################################################
                # win_size_2 = window_lists[9][block_num][i] ##################################################
                # win_size_3 = window_lists[10][block_num][i] ##################################################
                # win_size_4 = window_lists[11][block_num][i] ################################################## 

                win_size_1 = mask_list[8][block_num][i] ##################################################
                win_size_2 = mask_list[9][block_num][i] ##################################################
                win_size_3 = mask_list[10][block_num][i] ##################################################
                win_size_4 = mask_list[11][block_num][i] ##################################################

                attn_aftermask = attention_map
                # 倒数第一个mask
                # mask1 = torch.abs(x_cor1 - y_cor1) <= win_size_1
                # new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1

                # 倒数第二个mask              
                # mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size_2
                # new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask
                # mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size_3
                # new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:]
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3
                
                # 倒数第四个mask
                # mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) <= win_size_4
                # new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:]
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))
            out = attn_softmax @ value

        '''五个mask'''
        if layer == 7:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2]
            y3 = attention_map[:, :, :, left_edge3:right_edge3]
            y4 = attention_map[:, :, :, left_edge4:right_edge4]
            y5 = attention_map[:, :, :, left_edge5:right_edge5]
            
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            # grid_list = []
            # for i in range(layer):
            #     x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
            #     grid_list.append((x_cor, y_cor))
            # x_cor1,y_cor1 = grid_list[0]
            # x_cor2,y_cor2 = grid_list[1]
            # x_cor3,y_cor3 = grid_list[2]
            # x_cor4,y_cor4 = grid_list[3]
            # x_cor5,y_cor5 = grid_list[4]

            for i in range(y_mean1.shape[0]): # which head

                # 获取对应的 window 宽度
                # win_size_1 = window_lists[12][block_num][i] ##################################################
                # win_size_2 = window_lists[13][block_num][i]
                # win_size_3 = window_lists[14][block_num][i]
                # win_size_4 = window_lists[15][block_num][i]
                # win_size_5 = window_lists[16][block_num][i]

                win_size_1 = mask_list[12][block_num][i] ##################################################
                win_size_2 = mask_list[13][block_num][i]
                win_size_3 = mask_list[14][block_num][i]
                win_size_4 = mask_list[15][block_num][i]
                win_size_5 = mask_list[16][block_num][i]
            

                attn_aftermask = attention_map

                # 倒数第一个mask
                # mask1 = torch.abs(x_cor1 - y_cor1) <= win_size_1
                # new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                # mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size_2
                # new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                # mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size_3
                # new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:]
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                # mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) <= win_size_4
                # new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:]
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                # mask5 = torch.abs(x_cor5*(right_edge5-left_edge5)/edge - y_cor5) <= win_size_5
                # new_attn5 = mask5.unsqueeze(0) * y5[:,i,:,:]
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))
            out = attn_softmax @ value
            
        '''六个mask'''
        if layer == 8:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            y1 = attention_map[:, :, :, -edge:] # attentionmap(16,16,169,424),y1(16,16,169,169)
            y2 = attention_map[:, :, :, left_edge2:right_edge2]
            y3 = attention_map[:, :, :, left_edge3:right_edge3]
            y4 = attention_map[:, :, :, left_edge4:right_edge4]
            y5 = attention_map[:, :, :, left_edge5:right_edge5]
            y6 = attention_map[:, :, :, left_edge6:right_edge6]

            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            # grid_list = []
            # for i in range(layer):
            #     x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
            #     grid_list.append((x_cor, y_cor))
            # x_cor1,y_cor1 = grid_list[0]
            # x_cor2,y_cor2 = grid_list[1]
            # x_cor3,y_cor3 = grid_list[2]
            # x_cor4,y_cor4 = grid_list[3]
            # x_cor5,y_cor5 = grid_list[4]
            # x_cor6,y_cor6 = grid_list[5]

            for i in range(y_mean1.shape[0]): # which head
                # 获取对应的 window 宽度
                # win_size_1 = window_lists[17][block_num][i] ##################################################
                # win_size_2 = window_lists[18][block_num][i]
                # win_size_3 = window_lists[19][block_num][i]
                # win_size_4 = window_lists[20][block_num][i]
                # win_size_5 = window_lists[21][block_num][i]
                # win_size_6 = window_lists[22][block_num][i]

                win_size_1 = mask_list[17][block_num][i] ##################################################
                win_size_2 = mask_list[18][block_num][i]
                win_size_3 = mask_list[19][block_num][i]
                win_size_4 = mask_list[20][block_num][i]
                win_size_5 = mask_list[21][block_num][i]
                win_size_6 = mask_list[22][block_num][i]


                attn_aftermask = attention_map

                # 倒数第一个mask
                # mask1 = torch.abs(x_cor1 - y_cor1) <= win_size_1
                # new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]
                new_attn1 = win_size_1 * y1[:,i,:,:]
                attn_aftermask[:, i, :, -edge:] = new_attn1
                
                # 倒数第二个mask              
                # mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size_2
                # new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                new_attn2 = win_size_2 * y2[:,i,:,:]
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                # 倒数第三个mask              
                # mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size_3
                # new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:]
                new_attn3 = win_size_3 * y3[:,i,:,:]
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                # 倒数第四个mask              
                # mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) <= win_size_4
                # new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:]
                new_attn4 = win_size_4 * y4[:,i,:,:]
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                # 倒数第五个mask              
                # mask5 = torch.abs(x_cor5*(right_edge5-left_edge5)/edge - y_cor5) <= win_size_5
                # new_attn5 = mask5.unsqueeze(0) * y5[:,i,:,:]
                new_attn5 = win_size_5 * y5[:,i,:,:]
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                # 倒数第六个mask              
                # mask6 = torch.abs(x_cor6*(right_edge6-left_edge6)/edge - y_cor6) <= win_size_6
                # new_attn6 = mask6.unsqueeze(0) * y6[:,i,:,:]
                new_attn6 = win_size_6 * y6[:,i,:,:]
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                # 对mask后的attentionmap进行softmax
                attn_softmax = (F.dropout(attn_aftermask.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn_aftermask.softmax(dim=-1))
            out = attn_softmax @ value
    return out