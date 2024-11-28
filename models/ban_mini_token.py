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

import argparse
DEBUG=False

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

threshold_all = threshold
threshold_shallow = threshold
threshold_deep = threshold
# mini_threshold_2 = 0.2
# mini_threshold_3 = 0.1
# mini_threshold_4_1 = 0.1
# mini_threshold_4_2 = 0.05
# mini_threshold_4_3 = 0.05
# mini_threshold_4_4 = 0.05

# mini_threshold_5_1 = 0.05
# mini_threshold_5_2 = 0.05
# mini_threshold_5_3 = 0.05
# mini_threshold_5_4 = 0.05
# mini_threshold_5_5 = 0.03

# mini_threshold_6_1 = 0.03
# mini_threshold_6_2 = 0.03
# mini_threshold_6_3 = 0.03
# mini_threshold_6_4 = 0.03
# mini_threshold_6_5 = 0.03
# mini_threshold_6_6 = 0.02
mini_threshold_2 = 0.1
mini_threshold_3 = 0.05
mini_threshold_4_1 = 0.05
mini_threshold_4_2 = 0.05
mini_threshold_4_3 = 0.05
mini_threshold_4_4 = 0.03

mini_threshold_5_1 = 0.05
mini_threshold_5_2 = 0.05
mini_threshold_5_3 = 0.05
mini_threshold_5_4 = 0.05
mini_threshold_5_5 = 0.03

mini_threshold_6_1 = 0.03
mini_threshold_6_2 = 0.03
mini_threshold_6_3 = 0.03
mini_threshold_6_4 = 0.03
mini_threshold_6_5 = 0.03
mini_threshold_6_6 = 0.02
mini_threshold_7 = 0.05
# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None
block_cal_token_num_list_1 = {} # 创建一个空的字典来存储每个 block 的计数值和 mask后的token vs原本toke计数
block_cal_token_num_list_2 = {}
block_cal_token_num_list_3 = {}
block_cal_token_num_list_4 = {}
block_cal_token_num_list_5 = {}
block_cal_token_num_list_6 = {}
block_cal_token_num_list_7 = {}
window_wid_chosen_1 = {}
window_wid_chosen_2 = {}
window_wid_chosen_3 = {}
window_wid_chosen_4 = {}
window_wid_chosen_5 = {}
window_wid_chosen_6 = {}
window_wid_chosen_7 = {}
mask_chosen_save_1 = {}
mask_chosen_save_2 = {}
mask_chosen_save_3 = {}
mask_chosen_save_4 = {}
mask_chosen_save_5 = {}
mask_chosen_save_6 = {}
mask_chosen_save_7 = {}
def slow_attn_ban_mini_token_2(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    model_depth = query.shape[1]
    for counter in range(PIC_NUM):
        root_path = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/{model_depth}/threshold_{threshold}/temp/{counter}/'
        os.makedirs(root_path, exist_ok=True)
        file_path = os.path.join(root_path, 'mask_29.pth')
        if os.path.isfile(file_path):
            pass
        else:
            break

    # os.makedirs(root_path, exist_ok=True)
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    # # 对第一个维度进行分组操作 #####################################################################
    # for i in range(0, attn.shape[0] - 1, 2):
    #     attn[i] = attn[i + 1]

    if attn_mask is not None: attn.add_(attn_mask)
    # attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
    level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    for index, this_square in enumerate(token_scale):
        if query.shape[2] == this_square:
            layer = index
            break
    # torch.save(attention_map, root_path + f"attentionmap{ layer}.pth")
    if layer in range(0, 3):
        out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
    if layer in range(3,10):
        # attention_map = torch.load(f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/attention16/attentionmap{layer}.pth')
        attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
        # out = attention_map @ value
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
        # print(f'last mask is {attention_map[0, 0, :, -edge:].shape}')
        '''一个mask的情况:layer=9'''
        if layer == 9:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            cal_original_1 = edge*edge
            cal_original_2 = (right_edge2-left_edge2)*edge
            cal_original_3 = (right_edge3-left_edge3)*edge
            cal_original_4 = (right_edge4-left_edge4)*edge
            cal_original_5 = (right_edge5-left_edge5)*edge
            cal_original_6 = (right_edge6-left_edge6)*edge
            cal_original_7 = (right_edge7-left_edge7)*edge
            # 预先选定的模板窗口大小
            if query.shape[1] == 16:
                win_sizes1 = [16, 17, 18, 20, 30, 34, 37, 45, 47, 48, 50, 64, 75, 100, 125, 150, 175, 200, 225, 256] # 修改模板窗口大小
                win_sizes2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 169] # 修改模板窗口大小
                win_sizes3 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 85, 100] # 修改模板窗口大小
                win_sizes4 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 64] # 修改模板窗口大小
                win_sizes5 = list(range(5, 36, 2))
                win_sizes6 = list(range(5, 25, 2))
                win_sizes7 = list(range(5, 16, 2))
            elif query.shape[1] == 30:
                win_sizes1 = [16, 17, 32, 33, 34, 37, 48, 49, 50, 52, 54, 60, 64, 75, 100, 125, 150, 175, 200, 225, 256] # 修改模板窗口大小
                win_sizes2 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 169] # 修改模板窗口大小
                win_sizes3 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 85, 100] # 修改模板窗口大小
                win_sizes4 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 64] # 修改模板窗口大小
                win_sizes5 = list(range(5, 36, 2))
                win_sizes6 = list(range(5, 25, 2))
                win_sizes7 = list(range(5, 16, 2))
            y1_original = attention_map[:, :, :, -edge:].cpu() # attentionmap(16,16,169,424),y1(16,16,169,169)
            attention_original = attention_map.cpu()
            all_one_y1 = all_one[:, :, :, -edge:].cpu()
            y_mean1 = y1_original.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            # x_cor1, y_cor1 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer]))
            # x_cor2, y_cor2 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-1]))
            # x_cor3, y_cor3 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-2]))
            # x_cor4, y_cor4 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-3]))
            # x_cor5, y_cor5 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-4]))
            # x_cor6, y_cor6 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-5]))
            # x_cor7, y_cor7 = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-6]))
            grid_list = []
            for i in range(layer):
                x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
                grid_list.append((x_cor, y_cor))
            x_cor1,y_cor1 = grid_list[0]
            x_cor2,y_cor2 = grid_list[1]
            x_cor3,y_cor3 = grid_list[2]
            x_cor4,y_cor4 = grid_list[3]
            x_cor5,y_cor5 = grid_list[4]
            x_cor6,y_cor6 = grid_list[5]
            x_cor7,y_cor7 = grid_list[6]
            cal_rate_1 = []
            window_choose_1 = []
            cal_rate_2 = []
            window_choose_2 = []
            cal_rate_3 = []
            window_choose_3 = []
            cal_rate_4 = []
            window_choose_4 = []
            cal_rate_5 = []
            window_choose_5 = []
            cal_rate_6 = []
            window_choose_6 = []
            cal_rate_7 = []
            window_choose_7 = []
            mask_save_1 = []
            mask_save_2 = []
            mask_save_3 = []
            mask_save_4 = []
            mask_save_5 = []
            mask_save_6 = []
            mask_save_7 = []

            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            # '''更改：如果首两列加起来<0.1就先不抹了，这可能是个整张图都很小的情况'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            y4 = attention_map[:, :, :, left_edge4:right_edge4].cpu()
            y5 = attention_map[:, :, :, left_edge5:right_edge5].cpu()
            y6 = attention_map[:, :, :, left_edge6:right_edge6].cpu()
            y7 = attention_map[:, :, :, left_edge7:right_edge7].cpu()
            for i in range(y_mean1.shape[0]): # which head
                # print(f'HEAD {i}')
                '''排除有多条对角，导致最后一个对角占比不够的情况'''
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                window_3 = attention_map[:, i, :, left_edge3:right_edge3]
                window_4 = attention_map[:, i, :, left_edge4:right_edge4]
                window_5 = attention_map[:, i, :, left_edge5:right_edge5]
                window_6 = attention_map[:, i, :, left_edge6:right_edge6]
                window_7 = attention_map[:, i, :, left_edge7:right_edge7]
                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if ((window_2.sum() + window_3.sum())/attention_map[:,i,:,:].sum() < 0.2) and \
                    (y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < 0.2):
                # if attention_map[:, :, :, left_edge2:right_edge2]
                # if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_7: # 目前选为0.2
                    # print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # print(f'last 2 mask = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    # print(f'last 3 mask = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    # print(f'head = {i}')
                    # # 创建一个二维张量
                    # tensor = attention_map[0,i]
                    # print(tensor.shape)
                    # # 查找最大值及其对应的位置坐标
                    # max_value, max_indices = torch.max(tensor, dim=1)

                    # # 打印最大值
                    # print("最大值:", max_value)

                    # # 打印最大值对应的位置坐标
                    # print("最大值的位置坐标:", max_indices)
                    # # 遍历每一行
                    # for j in range(tensor.size(0)):
                    #     # 求取当前行的和
                    #     row_sum = torch.sum(tensor[j])
                        
                    #     # 打印当前行的和以及行索引
                    #     print(f"第{j+1}行的和: {row_sum}")

                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    mask_now = new_attn1[0].unsqueeze(0)*1.0
                    cal = 0
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                    mask_save_1.append(mask_now)################################
                    pass
                else:
                    # print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # print(f'last 2 mask = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    # print(f'last 3 mask = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    # # ''''''
                    
                    # print(f'head = {i}')
                    # # 创建一个二维张量
                    # tensor = attention_map[0,i]
                    # print(tensor.shape)
                    # # 查找最大值及其对应的位置坐标
                    # max_value, max_indices = torch.max(tensor, dim=1)

                    # # 打印最大值
                    # print("最大值:", max_value)

                    # # 打印最大值对应的位置坐标
                    # print("最大值的位置坐标:", max_indices)
                    # # 遍历每一行
                    # for j in range(tensor.size(0)):
                    #     # 求取当前行的和
                    #     row_sum = torch.sum(tensor[j])
                        
                    #     # 打印当前行的和以及行索引
                    #     print(f"第{j+1}行的和: {row_sum}")
                    
                    # # ''''''
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) < win_size # ([16, 16]),后续诸如([169, 169])
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_deep:
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                            mask_now = mask1.unsqueeze(0)*1.0
                            cal = mask1.sum()
                            cal_rate_1.append(int(cal))
                            window_choose_1.append(win_size)
                            mask_save_1.append(mask_now)
                            # print(f'HEAD {i} MASK 1 = {win_size}')
                            mask_tru1 += 1
                            break
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        mask_now = torch.ones_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                        # cal_original = token_scale[layer]*token_scale[layer]
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(256))
                        mask_save_1.append(mask_now)
                        # print(f'NO USE ANY MASK 1 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1
                '''IF MORE MASK IS VISIBLE'''
                '''如果存在第二对角'''
                if window_2.sum()/attention_map[:,i,:,:].sum() >= 0.1: # 认为存在多对角
                    # print('MORE MASK IS VISIBLE!!!!!!!!!!!!!!!!!!!!!!!!!')
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) < win_size
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:]
                            mask_now = mask2.unsqueeze(0)*1.0
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_save_2.append(mask_now)
                            mask_tru2 += 1
                            break
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        mask_now = torch.ones_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        mask_save_2.append(mask_now)
                else: 
                    '''如果不存在第二对角'''
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    mask_now = torch.zeros_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                    mask_save_2.append(mask_now)

                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                '''如果存在第三对角'''
                if window_3.sum()/attention_map[:,i,:,:].sum() >= 0.05: # 认为存在多对角
                    # 倒数第二个mask
                    mask_tru3 = 0
                    for win_size in win_sizes3:
                        mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) < win_size
                        attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:]
                            mask_now = mask3.unsqueeze(0)*1.0
                            cal3 = mask3.sum()
                            cal_rate_3.append(int(cal3))
                            window_choose_3.append(win_size)
                            mask_save_3.append(mask_now)
                            mask_tru3 += 1
                            break
                    if mask_tru3 == 0:
                        new_attn3 = y3[:,i,:,:]
                        mask_now = torch.ones_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_3.append(cal_original_3)
                        window_choose_3.append(int(right_edge3-left_edge3))
                        mask_save_3.append(mask_now)
                else: 
                    '''如果不存在第三对角'''
                    new_attn3 = torch.zeros_like(y3[:,i,:,:])
                    mask_now = torch.zeros_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_3.append(int(0))
                    window_choose_3.append(int(0))
                    mask_save_3.append(mask_now)
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                '''如果存在第四对角'''
                if window_4.sum()/attention_map[:,i,:,:].sum() >= 0.05: # 认为存在多对角
                    # 倒数第二个mask
                    mask_tru4 = 0
                    for win_size in win_sizes4:
                        mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) < win_size
                        attn_fraction = (mask4.unsqueeze(0) * y4[:,i,:,:]).sum() / y4[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:]
                            mask_now = mask4.unsqueeze(0)*1.0
                            cal4 = mask4.sum()
                            cal_rate_4.append(int(cal4))
                            window_choose_4.append(win_size)
                            mask_save_4.append(mask_now)
                            mask_tru4 += 1
                            break
                    if mask_tru4 == 0:
                        new_attn4 = y4[:,i,:,:]
                        mask_now = torch.ones_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_4.append(cal_original_4)
                        window_choose_4.append(int(right_edge4-left_edge4))
                        mask_save_4.append(mask_now)
                else: 
                    '''如果不存在第四对角'''
                    new_attn4 = torch.zeros_like(y4[:,i,:,:])
                    mask_now = torch.zeros_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_4.append(int(0))
                    window_choose_4.append(int(0))
                    mask_save_4.append(mask_now)
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                '''如果存在第五对角'''
                if window_5.sum()/attention_map[:,i,:,:].sum() >= 0.05: # 认为存在多对角
                    # 倒数第二个mask
                    mask_tru5 = 0
                    for win_size in win_sizes5:
                        mask5 = torch.abs(x_cor5*(right_edge5-left_edge5)/edge - y_cor5) < win_size
                        attn_fraction = (mask5.unsqueeze(0) * y5[:,i,:,:]).sum() / y5[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn5 = mask5.unsqueeze(0) * y5[:,i,:,:]
                            mask_now = mask5.unsqueeze(0)*1.0
                            cal5 = mask5.sum()
                            cal_rate_5.append(int(cal5))
                            window_choose_5.append(win_size)
                            mask_save_5.append(mask_now)
                            mask_tru5 += 1
                            break
                    if mask_tru5 == 0:
                        new_attn5 = y5[:,i,:,:]
                        mask_now = torch.ones_like(y5[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_5.append(cal_original_5)
                        window_choose_5.append(int(right_edge5-left_edge5))
                        mask_save_5.append(mask_now)
                else: 
                    '''如果不存在第五对角'''
                    new_attn5 = torch.zeros_like(y5[:,i,:,:])
                    mask_now = torch.zeros_like(y5[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_5.append(int(0))
                    window_choose_5.append(int(0))
                    mask_save_5.append(mask_now)
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                '''如果存在第六对角'''
                if window_6.sum()/attention_map[:,i,:,:].sum() >= 0.05: # 认为存在多对角
                    # 倒数第二个mask
                    mask_tru6 = 0
                    for win_size in win_sizes6:
                        mask6 = torch.abs(x_cor6*(right_edge6-left_edge6)/edge - y_cor6) < win_size
                        attn_fraction = (mask6.unsqueeze(0) * y6[:,i,:,:]).sum() / y6[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn6 = mask6.unsqueeze(0) * y6[:,i,:,:]
                            mask_now = mask6.unsqueeze(0)*1.0
                            cal6 = mask6.sum()
                            cal_rate_6.append(int(cal6))
                            window_choose_6.append(win_size)
                            mask_save_6.append(mask_now)
                            mask_tru6 += 1
                            break
                    if mask_tru6 == 0:
                        new_attn6 = y6[:,i,:,:]
                        mask_now = torch.ones_like(y6[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_6.append(cal_original_6)
                        window_choose_6.append(int(right_edge6-left_edge6))
                        mask_save_6.append(mask_now)
                else: 
                    '''如果不存在第六对角'''
                    new_attn6 = torch.zeros_like(y6[:,i,:,:])
                    mask_now = torch.zeros_like(y6[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_6.append(int(0))
                    window_choose_6.append(int(0))
                    mask_save_6.append(mask_now)
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

                '''如果存在第七对角'''
                if window_7.sum()/attention_map[:,i,:,:].sum() >= 0.05: # 认为存在多对角
                    # 倒数第二个mask
                    mask_tru7 = 0
                    for win_size in win_sizes7:
                        mask7 = torch.abs(x_cor7*(right_edge7-left_edge7)/edge - y_cor7) < win_size
                        attn_fraction = (mask7.unsqueeze(0) * y7[:,i,:,:]).sum() / y7[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn7 = mask7.unsqueeze(0) * y7[:,i,:,:]
                            mask_now = mask7.unsqueeze(0)*1.0
                            cal7 = mask7.sum()
                            cal_rate_7.append(int(cal7))
                            window_choose_7.append(win_size)
                            mask_save_7.append(mask_now)
                            mask_tru7 += 1
                            break
                    if mask_tru7 == 0:
                        new_attn7 = y7[:,i,:,:]
                        mask_now = torch.ones_like(y7[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_7.append(cal_original_7)
                        window_choose_7.append(int(right_edge7-left_edge7))
                        mask_save_7.append(mask_now)
                else: 
                    '''如果不存在第七对角'''
                    new_attn7 = torch.zeros_like(y7[:,i,:,:])
                    mask_now = torch.zeros_like(y7[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_7.append(int(0))
                    window_choose_7.append(int(0))
                    mask_save_7.append(mask_now)
                attn_aftermask[:, i, :, left_edge7:right_edge7] = new_attn7
            total_sum_oneblock_1 = sum([int(x) for x in cal_rate_1])
            total_sum_oneblock_2 = sum([int(x) for x in cal_rate_2])
            total_sum_oneblock_3 = sum([int(x) for x in cal_rate_3])
            total_sum_oneblock_4 = sum([int(x) for x in cal_rate_4])
            total_sum_oneblock_5 = sum([int(x) for x in cal_rate_5])
            total_sum_oneblock_6 = sum([int(x) for x in cal_rate_6])
            total_sum_oneblock_7 = sum([int(x) for x in cal_rate_7])
            block_cal_token_num_list_1[block_num] = total_sum_oneblock_1
            block_cal_token_num_list_2[block_num] = total_sum_oneblock_2
            block_cal_token_num_list_3[block_num] = total_sum_oneblock_3
            block_cal_token_num_list_4[block_num] = total_sum_oneblock_4
            block_cal_token_num_list_5[block_num] = total_sum_oneblock_5
            block_cal_token_num_list_6[block_num] = total_sum_oneblock_6
            block_cal_token_num_list_7[block_num] = total_sum_oneblock_7
            
            window_wid_chosen_1[block_num] = window_choose_1
            window_wid_chosen_2[block_num] = window_choose_2
            window_wid_chosen_3[block_num] = window_choose_3
            window_wid_chosen_4[block_num] = window_choose_4
            window_wid_chosen_5[block_num] = window_choose_5
            window_wid_chosen_6[block_num] = window_choose_6
            window_wid_chosen_7[block_num] = window_choose_7
            
            mask_chosen_save_1[block_num] = mask_save_1
            mask_chosen_save_2[block_num] = mask_save_2
            mask_chosen_save_3[block_num] = mask_save_3
            mask_chosen_save_4[block_num] = mask_save_4
            mask_chosen_save_5[block_num] = mask_save_5
            mask_chosen_save_6[block_num] = mask_save_6
            mask_chosen_save_7[block_num] = mask_save_7
                ##############################################################

            if block_num == model_depth-1:
                # 第一个对角
                # print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                # print(f'Mask we save: {mask_chosen_save_1}')
                file_name = f'mask_23.pth'
                torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                # 计算总计数
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                # 打印mask内总计数
                # print(f'Total calculation: {cal_token_aftermask_1}')
                # print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                # 第二个对角
                # print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                # print(f'Mask we save: {mask_chosen_save_2}')
                file_name = f'mask_24.pth'
                torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                # print(f'Total calculation: {cal_token_aftermask_2}')
                # print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                # 第三个对角
                # print('NOW FOR MASK 3    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_3}')
                # print(f'Mask we save: {mask_chosen_save_3}')
                file_name = f'mask_25.pth'
                torch.save(mask_chosen_save_3, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_3}')
                cal_token_aftermask_3 = sum(block_cal_token_num_list_3.values())
                # print(f'Total calculation: {cal_token_aftermask_3}')
                # print(f'Without mask: {cal_original_3*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_3/(cal_original_3*model_depth*model_depth)}')
                # 第四个对角
                # print('NOW FOR MASK 4    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_4}')
                # print(f'Mask we save: {mask_chosen_save_4}')
                file_name = f'mask_26.pth'
                torch.save(mask_chosen_save_4, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_4}')
                cal_token_aftermask_4 = sum(block_cal_token_num_list_4.values())
                # print(f'Total calculation: {cal_token_aftermask_4}')
                # print(f'Without mask: {cal_original_4*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_4/(cal_original_4*model_depth*model_depth)}')
                # 第五个对角
                # print('NOW FOR MASK 5    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_5}')
                # print(f'Mask we save: {mask_chosen_save_5}')
                file_name = f'mask_27.pth'
                torch.save(mask_chosen_save_5, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_5}')
                cal_token_aftermask_5 = sum(block_cal_token_num_list_5.values())
                # print(f'Total calculation: {cal_token_aftermask_5}')
                # print(f'Without mask: {cal_original_5*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_5/(cal_original_5*model_depth*model_depth)}')
                # 第六个对角
                # print('NOW FOR MASK 6    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_6}')
                # print(f'Mask we save: {mask_chosen_save_6}')
                file_name = f'mask_28.pth'
                torch.save(mask_chosen_save_6, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_6}')
                cal_token_aftermask_6 = sum(block_cal_token_num_list_6.values())
                # print(f'Total calculation: {cal_token_aftermask_6}')
                # print(f'Without mask: {cal_original_6*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_6/(cal_original_6*model_depth*model_depth)}')
                # 第七个对角
                # print('NOW FOR MASK 7    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_7}')
                # print(f'Mask we save: {mask_chosen_save_7}')
                file_name = f'mask_29.pth'
                torch.save(mask_chosen_save_7, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_7}')
                cal_token_aftermask_7 = sum(block_cal_token_num_list_7.values())
                # print(f'Total calculation: {cal_token_aftermask_7}')
                # print(f'Without mask: {cal_original_7*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_7/(cal_original_7*model_depth*model_depth)}')
                
                
                ############################
                # 打印全图总计数(目前对于mask外采用full-attn)
                before = left_edge7*edge
                # image_after_tokens = cal_token_aftermask_1+before*model_depth*model_depth
                image_after_tokens = (cal_token_aftermask_1+cal_token_aftermask_2+cal_token_aftermask_3+\
                                    cal_token_aftermask_4+cal_token_aftermask_5+cal_token_aftermask_6+\
                                        cal_token_aftermask_7)+before*model_depth*model_depth
                # image_before_tokens = cal_original_1*model_depth*model_depth+before*model_depth*model_depth
                image_before_tokens = (cal_original_1+cal_original_2+cal_original_3+cal_original_4+\
                                    cal_original_5+cal_original_6+cal_original_7+before)*model_depth*model_depth
                
                print(f'Image calculation after mask: {image_after_tokens}')
                print(f'Image calculation before mask: {image_before_tokens}')
                print(f'Image calculation rate: {(image_before_tokens-image_after_tokens)/image_before_tokens}')
                # 清空字典
                dict_list = [block_cal_token_num_list_1, block_cal_token_num_list_2, block_cal_token_num_list_3, \
                            block_cal_token_num_list_4, block_cal_token_num_list_5, block_cal_token_num_list_6, \
                                block_cal_token_num_list_7]
                window_list = [window_wid_chosen_1, window_wid_chosen_2, window_wid_chosen_3, window_wid_chosen_4, \
                            window_wid_chosen_5, window_wid_chosen_6, window_wid_chosen_7]

                for d, w in zip(dict_list, window_list):
                    d.clear()
                    w.clear()
                    # print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    # print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')
            
            out = attn_aftermask @ value
            # else:
            #     out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value

        '''两个mask的情况:layer=3'''
        if layer == 3:
            # if block_num == 0:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            cal_original_1 = edge*edge
            cal_original_2 = (right_edge2-left_edge2)*edge
            # 预先选定的模板窗口大小
            if query.shape[1] == 16:
                win_sizes1 = [5, 6, 7, 8, 9, 10, 11, 12, 15] # 修改模板窗口大小
                win_sizes2 = [4, 5, 6, 7, 9]
            elif query.shape[1] == 30:
                win_sizes1 = [5, 6, 7, 8, 9, 10, 11, 12, 13, 16] # 修改模板窗口大小
                win_sizes2 = [5, 6, 7, 8, 9]
            y1_original = attention_map[:, :, :, -edge:].cpu()
            y_mean1 = y1_original.mean(dim=0)
            grid_list = []
            for i in range(layer):
                x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
                grid_list.append((x_cor, y_cor))
            x_cor1,y_cor1 = grid_list[0]
            x_cor2,y_cor2 = grid_list[1]

            # x_cor1, y_cor1 = torch.meshgrid(torch.arange(y_mean1.shape[1]), torch.arange(y_mean1.shape[2]))
            # x_cor2, y_cor2 = torch.meshgrid(torch.arange(y_mean2.shape[1]), torch.arange(y_mean2.shape[2]))
            cal_rate_1 = []
            window_choose_1 = []
            cal_rate_2 = []
            window_choose_2 = []
            mask_save_1 = []
            mask_save_2 = []
            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            # '''更改：如果首两列加起来<0.1就先不抹了，这可能是个整张图都很小的情况'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            for i in range(y_mean1.shape[0]):
                # print(f'HEAD {i}')
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_2: # 目前选为0.2
                    # print(f'head = {i} mask 1 is skip')
                    # print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    mask_now = torch.zeros_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                    mask_save_1.append(mask_now)
                else:
                    # print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) < win_size # ([16, 16]),后续诸如([169, 169])
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                            mask_now = mask1.unsqueeze(0)*1.0
                            cal1 = mask1.sum()
                            cal_rate_1.append(int(cal1))
                            window_choose_1.append(win_size)
                            mask_save_1.append(mask_now)
                            mask_tru1 += 1
                            break
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        mask_now = torch.ones_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(edge))
                        mask_save_1.append(mask_now)

                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_2: # 目前选为0.2
                    # print(f'head = {i} mask 2 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    mask_now = torch.zeros_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                    mask_save_2.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    '''如果存在第二对角'''
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) < win_size
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                            mask_now = mask2.unsqueeze(0)*1.0
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_save_2.append(mask_now)
                            mask_tru2 += 1
                            break
                    
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        mask_now = torch.ones_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        mask_save_2.append(mask_now)
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

            total_sum_oneblock_1 = sum([int(x) for x in cal_rate_1])
            total_sum_oneblock_2 = sum([int(x) for x in cal_rate_2])

            block_cal_token_num_list_1[block_num] = total_sum_oneblock_1
            block_cal_token_num_list_2[block_num] = total_sum_oneblock_2

            window_wid_chosen_1[block_num] = window_choose_1
            window_wid_chosen_2[block_num] = window_choose_2
            
            mask_chosen_save_1[block_num] = mask_save_1
            mask_chosen_save_2[block_num] = mask_save_2

            if block_num == model_depth-1:
                # 第一个对角
                # print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                # print(f'Mask we save: {mask_chosen_save_1}')
                file_name = f'mask_0.pth'
                torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                # 计算总计数
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                # 打印mask内总计数
                # print(f'Total calculation: {cal_token_aftermask_1}')
                # print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                
                # 第二个对角
                # print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                # print(f'Mask we save: {mask_chosen_save_2}')
                file_name = f'mask_1.pth'
                torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                # 计算总计数
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                # 打印mask内总计数
                # print(f'Total calculation: {cal_token_aftermask_2}')
                # print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                

                ###################
                # 打印全图总计数
                before = left_edge2*edge
                image_after_tokens = (cal_token_aftermask_1+cal_token_aftermask_2)+before*model_depth*model_depth
                image_before_tokens = (cal_original_1+cal_original_2+before)*model_depth*model_depth
                print(f'Image calculation after mask: {image_after_tokens}')
                print(f'Image calculation before mask: {image_before_tokens}')
                print(f'Image calculation rate: {(image_before_tokens-image_after_tokens)/image_before_tokens}')

                # 清除字典
                dict_list = [block_cal_token_num_list_1, block_cal_token_num_list_2]
                window_list = [window_wid_chosen_1, window_wid_chosen_2]

                for d, w in zip(dict_list, window_list):
                    d.clear()
                    w.clear()
                    # print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    # print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')

            out = attn_aftermask @ value
            # else:
            #     out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
        
        '''三个mask的情况:layer=4和layer=5'''
        if layer == 4 or layer == 5:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            cal_original_1 = edge*edge
            cal_original_2 = (right_edge2-left_edge2)*edge
            cal_original_3 = (right_edge3-left_edge3)*edge
            if layer == 4:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 25]  # 修改模板窗口大小
                    win_sizes2 = [6, 7, 8, 9, 10, 11, 13, 15]
                    win_sizes3 = [4, 5, 6, 7, 9]
                elif query.shape[1] == 30:
                    win_sizes1 = [6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25] # 修改模板窗口大小
                    win_sizes2 = [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
                    win_sizes3 = [5, 6, 7, 8, 9]
            elif layer == 5:
                # 预先选定的模板窗口大小
                if query.shape[1] == 16:
                    win_sizes1 = [6, 7, 11, 12, 13, 18, 20, 25, 28, 30]  
                    win_sizes2 = [6, 7, 8, 9, 10, 13, 15, 17, 20, 25]
                    win_sizes3 = [5, 6, 7, 8, 10, 12, 13]
                elif query.shape[1] == 30:
                    win_sizes1 = [6, 8, 10, 12, 13, 16, 18, 22, 24, 26, 28, 30, 35, 36] # 修改模板窗口大小
                    win_sizes2 = [6, 8, 10, 11, 12, 13, 14, 16, 17, 18, 20, 25]
                    win_sizes3 = [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]
            
            y1 = attention_map[:, :, :, -edge:].cpu() # attentionmap(16,16,169,424),y1(16,16,169,169)
            y_mean1 = y1.mean(dim=0) # torch.Size([=16, ……, ……]) 第一维直接求平均，因为观察发现第一维结果基本都一样
            grid_list = []
            for i in range(layer):
                x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
                grid_list.append((x_cor, y_cor))
            x_cor1,y_cor1 = grid_list[0]
            x_cor2,y_cor2 = grid_list[1]
            x_cor3,y_cor3 = grid_list[2]
            cal_rate_1 = []
            window_choose_1 = []
            cal_rate_2 = []
            window_choose_2 = []
            cal_rate_3 = []
            window_choose_3 = []

            mask_save_1 = []
            mask_save_2 = []
            mask_save_3 = []
            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            # '''更改：如果首两列加起来<0.1就先不抹了，这可能是个整张图都很小的情况'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            for i in range(y_mean1.shape[0]):
                # print(f'HEAD {i}')
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                window_3 = attention_map[:, i, :, left_edge3:right_edge3]
                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_3: # 目前选为0.2
                    # print(f'head = {i} mask 1 is skip')
                    # print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    mask_now = torch.zeros_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                    mask_save_1.append(mask_now)
                else:
                    # print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) < win_size # ([16, 16]),后续诸如([169, 169])
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                            mask_now = mask1.unsqueeze(0)*1.0
                            cal1 = mask1.sum()
                            cal_rate_1.append(int(cal1))
                            window_choose_1.append(win_size)
                            mask_save_1.append(mask_now)
                            mask_tru1 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        mask_now = torch.ones_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(edge))
                        mask_save_1.append(mask_now)
                        # print(f'NO USE ANY MASK 1 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_3: # 目前选为0.2
                    # print(f'head = {i} mask 2 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    mask_now = torch.zeros_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                    mask_save_2.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    '''如果存在第二对角'''
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) < win_size
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                            mask_now = mask2.unsqueeze(0)*1.0
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_save_2.append(mask_now)
                            mask_tru2 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        mask_now = torch.ones_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        mask_save_2.append(mask_now)
                        # print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_3: # 目前选为0.1
                    # print(f'head = {i} mask 3 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn3 = torch.zeros_like(y3[:,i,:,:])
                    mask_now = torch.zeros_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_3.append(int(0))
                    window_choose_3.append(int(0))
                    mask_save_3.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    '''如果存在第三对角'''
                    # 倒数第三个mask
                    mask_tru3 = 0
                    for win_size in win_sizes3:
                        mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) < win_size
                        # attn_fraction = (mask3 * y_mean3[i]).sum() / y_mean3[i].sum()
                        attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                            mask_now = mask3.unsqueeze(0)*1.0
                            cal3 = mask3.sum()
                            cal_rate_3.append(int(cal3))
                            window_choose_3.append(win_size)
                            mask_save_3.append(mask_now)
                            mask_tru3 += 1
                            break
                    if mask_tru3 == 0:
                        new_attn3 = y3[:,i,:,:]
                        mask_now = torch.ones_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_3.append(cal_original_3)
                        window_choose_3.append(int(right_edge3-left_edge3))
                        mask_save_3.append(mask_now)
                        # print(f'NO USE ANY MASK 3 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3
            # print(F'Attn_aftermask.shape={attn_aftermask.shape}')
            total_sum_oneblock_1 = sum([int(x) for x in cal_rate_1])
            total_sum_oneblock_2 = sum([int(x) for x in cal_rate_2])
            total_sum_oneblock_3 = sum([int(x) for x in cal_rate_3])

            block_cal_token_num_list_1[block_num] = total_sum_oneblock_1
            block_cal_token_num_list_2[block_num] = total_sum_oneblock_2
            block_cal_token_num_list_3[block_num] = total_sum_oneblock_3

            window_wid_chosen_1[block_num] = window_choose_1
            window_wid_chosen_2[block_num] = window_choose_2
            window_wid_chosen_3[block_num] = window_choose_3

            mask_chosen_save_1[block_num] = mask_save_1
            mask_chosen_save_2[block_num] = mask_save_2
            mask_chosen_save_3[block_num] = mask_save_3

            if block_num == model_depth-1:
                # 第一个对角
                # print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                # print(f'Mask we save: {mask_chosen_save_1}')
                if layer == 4:
                    file_name = f'mask_2.pth'
                    torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                elif layer == 5:
                    file_name = f'mask_5.pth'
                    torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                # print(f'Total calculation: {cal_token_aftermask_1}')
                # print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                
                # 第二个对角
                # print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                # print(f'Mask we save: {mask_chosen_save_2}')
                if layer == 4:
                    file_name = f'mask_3.pth'
                    torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                elif layer == 5:
                    file_name = f'mask_6.pth'
                    torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                # print(f'Total calculation: {cal_token_aftermask_2}')
                # print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                
                # 第三个对角
                # print('NOW FOR MASK 3    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_3}')
                # print(f'Mask we save: {mask_chosen_save_3}')
                if layer == 4:
                    file_name = f'mask_4.pth'
                    torch.save(mask_chosen_save_3, os.path.join(root_path, file_name)) # 保存 Mask we save
                elif layer == 5:
                    file_name = f'mask_7.pth'
                    torch.save(mask_chosen_save_3, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_3}')
                cal_token_aftermask_3 = sum(block_cal_token_num_list_3.values())
                # print(f'Total calculation: {cal_token_aftermask_3}')
                # print(f'Without mask: {cal_original_3*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_3/(cal_original_3*model_depth*model_depth)}')
                
                ###################
                # 打印全图总计数
                before = left_edge3*edge
                image_after_tokens = (cal_token_aftermask_1+cal_token_aftermask_2+cal_token_aftermask_3)+before*model_depth*model_depth
                image_before_tokens = (cal_original_1+cal_original_2+cal_original_3+before)*model_depth*model_depth
                print(f'Image calculation after mask: {image_after_tokens}')
                print(f'Image calculation before mask: {image_before_tokens}')
                print(f'Image calculation rate: {(image_before_tokens-image_after_tokens)/image_before_tokens}')

                # 清除字典
                dict_list = [block_cal_token_num_list_1, block_cal_token_num_list_2, block_cal_token_num_list_3]
                window_list = [window_wid_chosen_1, window_wid_chosen_2, window_wid_chosen_3]

                for d, w in zip(dict_list, window_list):
                    d.clear()
                    w.clear()
                    # print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    # print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')

            out = attn_aftermask @ value

        '''四个mask的情况:layer=6'''
        if layer == 6:
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            # out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
            cal_original_1 = edge*edge
            cal_original_2 = (right_edge2-left_edge2)*edge
            cal_original_3 = (right_edge3-left_edge3)*edge
            cal_original_4 = (right_edge4-left_edge4)*edge
            if query.shape[1] == 16:
                win_sizes1 = [8, 9, 11, 16, 18, 26, 30, 43, 51, 64]  # 修改模板窗口大小
                win_sizes2 = [8, 9, 10, 11, 12, 13, 17, 24, 30, 35]
                win_sizes3 = [6, 7, 8, 9, 10, 15, 20, 25]
                win_sizes4 = [6, 7, 9, 11, 13, 16]
            elif query.shape[1] == 30:
                win_sizes1 = [8, 9, 10, 13, 15, 16, 17, 20, 23, 24, 26, 40, 41, 42, 43, 44, 46, 48, 50, 64]  # 修改模板窗口大小
                win_sizes2 = [5, 9, 10, 11, 13, 14, 16, 17, 23, 24, 25, 26, 27, 29, 30, 35]
                win_sizes3 = [7, 8, 9, 11, 13, 14, 15, 16, 17, 18, 20, 25]
                win_sizes4 = [5, 7, 9, 11, 12, 13, 16]

            y1 = attention_map[:, :, :, -edge:].cpu()
            y_mean1 = y1.mean(dim=0)
            grid_list = []
            for i in range(layer):
                x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
                grid_list.append((x_cor, y_cor))
            x_cor1,y_cor1 = grid_list[0]
            x_cor2,y_cor2 = grid_list[1]
            x_cor3,y_cor3 = grid_list[2]
            x_cor4,y_cor4 = grid_list[3]
            cal_rate_1 = []
            window_choose_1 = []
            cal_rate_2 = []
            window_choose_2 = []
            cal_rate_3 = []
            window_choose_3 = []
            cal_rate_4 = []
            window_choose_4 = []

            mask_save_1 = []
            mask_save_2 = []
            mask_save_3 = []
            mask_save_4 = []

            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            y4 = attention_map[:, :, :, left_edge4:right_edge4].cpu()
            for i in range(y_mean1.shape[0]):
                # print(f'HEAD {i}')
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                window_3 = attention_map[:, i, :, left_edge3:right_edge3]
                window_4 = attention_map[:, i, :, left_edge4:right_edge4]
                # print(f'last 1 mask = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 2 mask = {y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 3 mask = {y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 4 mask = {y4[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_4_1: # 目前选为0.1
                    # print(f'head = {i} mask 1 is skip')
                    # print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    mask_now = torch.zeros_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                    mask_save_1.append(mask_now)
                else:
                    # print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) < win_size # ([16, 16]),后续诸如([169, 169])
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_deep:
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                            mask_now = mask1.unsqueeze(0)*1.0
                            cal1 = mask1.sum()
                            cal_rate_1.append(int(cal1))
                            window_choose_1.append(win_size)
                            mask_save_1.append(mask_now)
                            mask_tru1 += 1
                            break
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        mask_now = torch.ones_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(edge))
                        mask_save_1.append(mask_now)
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_4_2: # 目前选为0.1
                    # print(f'head = {i} mask 2 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    mask_now = torch.zeros_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                    mask_save_2.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    '''第二对角'''
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) < win_size
                        # attn_fraction = (mask2 * y_mean2[i]).sum() / y_mean2[i].sum()
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            # new_attn1 = mask1 * y1
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                            mask_now = mask2.unsqueeze(0)*1.0
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_save_2.append(mask_now)
                            # print(f'HEAD {i} MASK 2 = {win_size}')
                            mask_tru2 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        mask_now = torch.ones_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        mask_save_2.append(mask_now)
                        # print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_4_3: # 目前选为0.1
                    # print(f'head = {i} mask 3 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn3 = torch.zeros_like(y3[:,i,:,:])
                    mask_now = torch.zeros_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_3.append(int(0))
                    window_choose_3.append(int(0))
                    mask_save_3.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    '''第三对角'''
                    # 倒数第三个mask
                    mask_tru3 = 0
                    for win_size in win_sizes3:
                        mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) < win_size
                        attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                            mask_now = mask3.unsqueeze(0)*1.0
                            cal3 = mask3.sum()
                            cal_rate_3.append(int(cal3))
                            window_choose_3.append(win_size)
                            mask_save_3.append(mask_now)
                            mask_tru3 += 1
                            break
                    if mask_tru3 == 0:
                        new_attn3 = y3[:,i,:,:]
                        mask_now = torch.ones_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_3.append(cal_original_3)
                        window_choose_3.append(int(right_edge3-left_edge3))
                        mask_save_3.append(mask_now)
                        # print(f'NO USE ANY MASK 3 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y4[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_4_4: # 目前选为0.1
                    # print(f'head = {i} mask 4 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge4:right_edge4].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn4 = torch.zeros_like(y4[:,i,:,:])
                    mask_now = torch.zeros_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_4.append(int(0))
                    window_choose_4.append(int(0))
                    mask_save_4.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge4:right_edge4].sum()/attention_map[:,i,:,:].sum()}')
                    '''第四对角'''
                    # 倒数第四个mask
                    mask_tru4 = 0
                    for win_size in win_sizes4:
                        mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) < win_size
                        attn_fraction = (mask4.unsqueeze(0) * y4[:,i,:,:]).sum() / y4[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:] #
                            mask_now = mask4.unsqueeze(0)*1.0
                            cal4 = mask4.sum()
                            cal_rate_4.append(int(cal4))
                            window_choose_4.append(win_size)
                            mask_save_4.append(mask_now)
                            mask_tru4 += 1
                            break
                    if mask_tru4 == 0:
                        new_attn4 = y4[:,i,:,:]
                        mask_now = torch.ones_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_4.append(cal_original_4)
                        window_choose_4.append(int(right_edge4-left_edge4))
                        mask_save_4.append(mask_now)
                        # print(f'NO USE ANY MASK 4 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

            total_sum_oneblock_1 = sum([int(x) for x in cal_rate_1])
            total_sum_oneblock_2 = sum([int(x) for x in cal_rate_2])
            total_sum_oneblock_3 = sum([int(x) for x in cal_rate_3])
            total_sum_oneblock_4 = sum([int(x) for x in cal_rate_4])

            block_cal_token_num_list_1[block_num] = total_sum_oneblock_1
            block_cal_token_num_list_2[block_num] = total_sum_oneblock_2
            block_cal_token_num_list_3[block_num] = total_sum_oneblock_3
            block_cal_token_num_list_4[block_num] = total_sum_oneblock_4

            window_wid_chosen_1[block_num] = window_choose_1
            window_wid_chosen_2[block_num] = window_choose_2
            window_wid_chosen_3[block_num] = window_choose_3
            window_wid_chosen_4[block_num] = window_choose_4

            mask_chosen_save_1[block_num] = mask_save_1
            mask_chosen_save_2[block_num] = mask_save_2
            mask_chosen_save_3[block_num] = mask_save_3
            mask_chosen_save_4[block_num] = mask_save_4

            if block_num == model_depth-1:
                # 第一个对角
                # print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                # print(f'Mask we save: {mask_chosen_save_1}')
                file_name = f'mask_8.pth'
                torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                # print(f'Total calculation: {cal_token_aftermask_1}')
                # print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                
                # 第二个对角
                # print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                # print(f'Mask we save: {mask_chosen_save_2}')
                file_name = f'mask_9.pth'
                torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                # print(f'Total calculation: {cal_token_aftermask_2}')
                # print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                
                # 第三个对角
                # print('NOW FOR MASK 3    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_3}')
                # print(f'Mask we save: {mask_chosen_save_3}')
                file_name = f'mask_10.pth'
                torch.save(mask_chosen_save_3, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_3}')
                cal_token_aftermask_3 = sum(block_cal_token_num_list_3.values())
                # print(f'Total calculation: {cal_token_aftermask_3}')
                # print(f'Without mask: {cal_original_3*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_3/(cal_original_3*model_depth*model_depth)}')
                
                # 第四个对角
                # print('NOW FOR MASK 4    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_4}')
                # print(f'Mask we save: {mask_chosen_save_4}')
                file_name = f'mask_11.pth'
                torch.save(mask_chosen_save_4, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_4}')
                cal_token_aftermask_4 = sum(block_cal_token_num_list_4.values())
                # print(f'Total calculation: {cal_token_aftermask_4}')
                # print(f'Without mask: {cal_original_4*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_4/(cal_original_4*model_depth*model_depth)}')
                
                ###################
                # 打印全图总计数
                before = left_edge4*edge
                image_after_tokens = (cal_token_aftermask_1+cal_token_aftermask_2+\
                                      cal_token_aftermask_3+cal_token_aftermask_4)+before*model_depth*model_depth
                image_before_tokens = (cal_original_1+cal_original_2+cal_original_3+cal_original_4+before)*model_depth*model_depth
                print(f'Image calculation after mask: {image_after_tokens}')
                print(f'Image calculation before mask: {image_before_tokens}')
                print(f'Image calculation rate: {(image_before_tokens-image_after_tokens)/image_before_tokens}')

                # 清除字典
                dict_list = [block_cal_token_num_list_1, block_cal_token_num_list_2, block_cal_token_num_list_3,\
                             block_cal_token_num_list_4]
                window_list = [window_wid_chosen_1, window_wid_chosen_2, window_wid_chosen_3,\
                               window_wid_chosen_4]

                for d, w in zip(dict_list, window_list):
                    d.clear()
                    w.clear()
                    # print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    # print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')

            out = attn_aftermask @ value

        if layer == 7: # 5 个mask
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            cal_original_1 = edge*edge
            cal_original_2 = (right_edge2-left_edge2)*edge
            cal_original_3 = (right_edge3-left_edge3)*edge
            cal_original_4 = (right_edge4-left_edge4)*edge
            cal_original_5 = (right_edge5-left_edge5)*edge
            # 预先选定的模板窗口大小
            if query.shape[1] == 16:
                win_sizes1 = [11, 20, 21, 22, 25, 30, 50, 64, 75, 85, 90, 100]
                win_sizes2 = [9, 10, 13, 18, 20, 31, 46, 51, 64]
                win_sizes3 = [7, 9, 10, 12, 20, 25, 30, 36]
                win_sizes4 = [7, 8, 9, 10, 11, 15, 18, 20, 25]
                win_sizes5 = [6, 7, 9, 11, 13, 16]
            elif query.shape[1] == 30:
                win_sizes1 = [11, 20, 22, 25, 28, 30, 32, 35, 37, 40, 51, 56, 60, 64, 75, 85, 90, 100]  # 修改模板窗口大小
                win_sizes2 = [8, 10, 13, 15, 16, 18, 20, 22, 24, 25, 30, 32, 38, 40, 42, 43, 44, 46, 48, 49, 50, 51, 64]
                win_sizes3 = [6, 9, 10, 11, 13, 14, 17, 19, 21, 23, 25, 27, 28, 29, 35]
                win_sizes4 = [5, 6, 8, 11, 12, 14, 15, 17, 19, 20, 25]
                win_sizes5 = [5, 7, 9, 11, 12, 13, 16]
            y1 = attention_map[:, :, :, -edge:].cpu()
            y_mean1 = y1.mean(dim=0)
            grid_list = []
            for i in range(layer):
                x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
                grid_list.append((x_cor, y_cor))
            x_cor1,y_cor1 = grid_list[0]
            x_cor2,y_cor2 = grid_list[1]
            x_cor3,y_cor3 = grid_list[2]
            x_cor4,y_cor4 = grid_list[3]
            x_cor5,y_cor5 = grid_list[4]
            cal_rate_1 = []
            window_choose_1 = []
            cal_rate_2 = []
            window_choose_2 = []
            cal_rate_3 = []
            window_choose_3 = []
            cal_rate_4 = []
            window_choose_4 = []
            cal_rate_5 = []
            window_choose_5 = []

            mask_save_1 = []
            mask_save_2 = []
            mask_save_3 = []
            mask_save_4 = []
            mask_save_5 = []

            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            y4 = attention_map[:, :, :, left_edge4:right_edge4].cpu()
            y5 = attention_map[:, :, :, left_edge5:right_edge5].cpu()
            for i in range(y_mean1.shape[0]):
                # print(f'BLOCK {block_num}')
                # print(f'HEAD {i}')
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                window_3 = attention_map[:, i, :, left_edge3:right_edge3]
                window_4 = attention_map[:, i, :, left_edge4:right_edge4]
                window_5 = attention_map[:, i, :, left_edge5:right_edge5]
                # print(f'last 1 mask = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 2 mask = {y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 3 mask = {y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 4 mask = {y4[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 5 mask = {y5[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_5_1: # 目前选为
                    # print(f'head = {i} mask 1 is skip')
                    # print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    mask_now = torch.zeros_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                    mask_save_1.append(mask_now)
                else:
                    # print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) < win_size
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_deep:
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]
                            mask_now = mask1.unsqueeze(0)*1.0
                            cal1 = mask1.sum()
                            cal_rate_1.append(int(cal1))
                            window_choose_1.append(win_size)
                            mask_save_1.append(mask_now)
                            mask_tru1 += 1
                            break
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        mask_now = torch.ones_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(edge))
                        mask_save_1.append(mask_now)
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_5_2: # 目前选为
                    # print(f'head = {i} mask 2 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    mask_now = torch.zeros_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                    mask_save_2.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    '''第二对角'''
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) < win_size
                        # attn_fraction = (mask2 * y_mean2[i]).sum() / y_mean2[i].sum()
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            # new_attn1 = mask1 * y1
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                            mask_now = mask2.unsqueeze(0)*1.0
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_save_2.append(mask_now)
                            # print(f'HEAD {i} MASK 2 = {win_size}')
                            mask_tru2 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        mask_now = torch.ones_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        mask_save_2.append(mask_now)
                        # print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_5_3: # 目前选为0.1
                    # print(f'head = {i} mask 3 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn3 = torch.zeros_like(y3[:,i,:,:])
                    mask_now = torch.zeros_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_3.append(int(0))
                    window_choose_3.append(int(0))
                    mask_save_3.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    '''第三对角'''
                    # 倒数第三个mask
                    mask_tru3 = 0
                    for win_size in win_sizes3:
                        mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) < win_size
                        # attn_fraction = (mask3 * y_mean3[i]).sum() / y_mean3[i].sum()
                        attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                            mask_now = mask3.unsqueeze(0)*1.0
                            # print(f'HEAD {i} MASK 3 = {win_size}')
                            cal3 = mask3.sum()
                            cal_rate_3.append(int(cal3))
                            window_choose_3.append(win_size)
                            mask_save_3.append(mask_now)
                            mask_tru3 += 1
                            break
                    if mask_tru3 == 0:
                        new_attn3 = y3[:,i,:,:]
                        mask_now = torch.ones_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_3.append(cal_original_3)
                        window_choose_3.append(int(right_edge3-left_edge3))
                        mask_save_3.append(mask_now)
                        # print(f'NO USE ANY MASK 3 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y4[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_5_4: # 目前选为
                    # print(f'head = {i} mask 4 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge4:right_edge4].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn4 = torch.zeros_like(y4[:,i,:,:])
                    mask_now = torch.zeros_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_4.append(int(0))
                    window_choose_4.append(int(0))
                    mask_save_4.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge4:right_edge4].sum()/attention_map[:,i,:,:].sum()}')
                    '''第四对角'''
                    # 倒数第四个mask
                    mask_tru4 = 0
                    for win_size in win_sizes4:
                        mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) < win_size
                        # attn_fraction = (mask4 * y_mean4[i]).sum() / y_mean4[i].sum()
                        attn_fraction = (mask4.unsqueeze(0) * y4[:,i,:,:]).sum() / y4[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:] #
                            mask_now = mask4.unsqueeze(0)*1.0
                            # print(f'HEAD {i} MASK 4 = {win_size}')
                            cal4 = mask4.sum()
                            cal_rate_4.append(int(cal4))
                            window_choose_4.append(win_size)
                            mask_save_4.append(mask_now)
                            mask_tru4 += 1
                            break
                    if mask_tru4 == 0:
                        new_attn4 = y4[:,i,:,:]
                        mask_now = torch.ones_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_4.append(cal_original_4)
                        window_choose_4.append(int(right_edge4-left_edge4))
                        mask_save_4.append(mask_now)
                        # print(f'NO USE ANY MASK 4 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y5[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_5_5: # 目前选为
                    # print(f'head = {i} mask 5 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge5:right_edge5].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn5 = torch.zeros_like(y5[:,i,:,:])
                    mask_now = torch.zeros_like(y5[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_5.append(int(0))
                    window_choose_5.append(int(0))
                    mask_save_5.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge5:right_edge5].sum()/attention_map[:,i,:,:].sum()}')
                    '''第五对角'''
                    # 倒数第五个mask
                    mask_tru5 = 0
                    for win_size in win_sizes5:
                        mask5 = torch.abs(x_cor5*(right_edge5-left_edge5)/edge - y_cor5) < win_size
                        attn_fraction = (mask5.unsqueeze(0) * y5[:,i,:,:]).sum() / y5[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn5 = mask5.unsqueeze(0) * y5[:,i,:,:]
                            mask_now = mask5.unsqueeze(0)*1.0
                            cal5 = mask5.sum()
                            cal_rate_5.append(int(cal5))
                            window_choose_5.append(win_size)
                            mask_save_5.append(mask_now)
                            mask_tru5 += 1
                            break
                    if mask_tru5 == 0:
                        new_attn5 = y5[:,i,:,:]
                        mask_now = torch.ones_like(y5[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_5.append(cal_original_5)
                        window_choose_5.append(int(right_edge5-left_edge5))
                        mask_save_5.append(mask_now)
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5
            total_sum_oneblock_1 = sum([int(x) for x in cal_rate_1])
            total_sum_oneblock_2 = sum([int(x) for x in cal_rate_2])
            total_sum_oneblock_3 = sum([int(x) for x in cal_rate_3])
            total_sum_oneblock_4 = sum([int(x) for x in cal_rate_4])
            total_sum_oneblock_5 = sum([int(x) for x in cal_rate_5])

            block_cal_token_num_list_1[block_num] = total_sum_oneblock_1
            block_cal_token_num_list_2[block_num] = total_sum_oneblock_2
            block_cal_token_num_list_3[block_num] = total_sum_oneblock_3
            block_cal_token_num_list_4[block_num] = total_sum_oneblock_4
            block_cal_token_num_list_5[block_num] = total_sum_oneblock_5

            window_wid_chosen_1[block_num] = window_choose_1
            window_wid_chosen_2[block_num] = window_choose_2
            window_wid_chosen_3[block_num] = window_choose_3
            window_wid_chosen_4[block_num] = window_choose_4
            window_wid_chosen_5[block_num] = window_choose_5

            mask_chosen_save_1[block_num] = mask_save_1
            mask_chosen_save_2[block_num] = mask_save_2
            mask_chosen_save_3[block_num] = mask_save_3
            mask_chosen_save_4[block_num] = mask_save_4
            mask_chosen_save_5[block_num] = mask_save_5

            if block_num == model_depth-1:
                # 第一个对角
                # print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                # print(f'Mask we save: {mask_chosen_save_1}')
                file_name = f'mask_12.pth'
                torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                # print(f'Total calculation: {cal_token_aftermask_1}')
                # print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                
                # 第二个对角
                # print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                # print(f'Mask we save: {mask_chosen_save_2}')
                file_name = f'mask_13.pth'
                torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                # print(f'Total calculation: {cal_token_aftermask_2}')
                # print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                
                # 第三个对角
                # print('NOW FOR MASK 3    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_3}')
                # print(f'Mask we save: {mask_chosen_save_3}')
                file_name = f'mask_14.pth'
                torch.save(mask_chosen_save_3, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_3}')
                cal_token_aftermask_3 = sum(block_cal_token_num_list_3.values())
                # print(f'Total calculation: {cal_token_aftermask_3}')
                # print(f'Without mask: {cal_original_3*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_3/(cal_original_3*model_depth*model_depth)}')
                
                # 第四个对角
                # print('NOW FOR MASK 4    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_4}')
                # print(f'Mask we save: {mask_chosen_save_4}')
                file_name = f'mask_15.pth'
                torch.save(mask_chosen_save_4, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_4}')
                cal_token_aftermask_4 = sum(block_cal_token_num_list_4.values())
                # print(f'Total calculation: {cal_token_aftermask_4}')
                # print(f'Without mask: {cal_original_4*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_4/(cal_original_4*model_depth*model_depth)}')

                # 第五个对角
                # print('NOW FOR MASK 5    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_5}')
                # print(f'Mask we save: {mask_chosen_save_5}')
                file_name = f'mask_16.pth'
                torch.save(mask_chosen_save_5, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_5}')
                cal_token_aftermask_5 = sum(block_cal_token_num_list_5.values())
                # print(f'Total calculation: {cal_token_aftermask_5}')
                # print(f'Without mask: {cal_original_5*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_5/(cal_original_5*model_depth*model_depth)}')
                
                ###################
                # 打印全图总计数
                before = left_edge5*edge
                image_after_tokens = (cal_token_aftermask_1+cal_token_aftermask_2+cal_token_aftermask_3+\
                                      cal_token_aftermask_4+cal_token_aftermask_5)+before*model_depth*model_depth
                image_before_tokens = (cal_original_1+cal_original_2+cal_original_3+\
                                       cal_original_4+cal_original_5+before)*model_depth*model_depth
                print(f'Image calculation after mask: {image_after_tokens}')
                print(f'Image calculation before mask: {image_before_tokens}')
                print(f'Image calculation rate: {(image_before_tokens-image_after_tokens)/image_before_tokens}')

                # 清除字典
                dict_list = [block_cal_token_num_list_1, block_cal_token_num_list_2, block_cal_token_num_list_3,\
                             block_cal_token_num_list_4, block_cal_token_num_list_5]
                window_list = [window_wid_chosen_1, window_wid_chosen_2, window_wid_chosen_3,\
                               window_wid_chosen_4, window_wid_chosen_5]

                for d, w in zip(dict_list, window_list):
                    d.clear()
                    w.clear()
                    # print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    # print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')

            out = attn_aftermask @ value



        if layer == 8:    # 六个mask
            # print(f'##################    BLOCK {block_num}    ######################')
            # print(f'##################    SCALE {layer}    ######################')
            cal_original_1 = edge*edge
            cal_original_2 = (right_edge2-left_edge2)*edge
            cal_original_3 = (right_edge3-left_edge3)*edge
            cal_original_4 = (right_edge4-left_edge4)*edge
            cal_original_5 = (right_edge5-left_edge5)*edge
            cal_original_6 = (right_edge6-left_edge6)*edge
            if query.shape[1] == 16:
                win_sizes1 = [14, 15, 28, 35, 39, 42, 50, 57, 64, 70, 85, 100, 125, 150, 169]
                win_sizes2 = [11, 12, 20, 23, 24, 30, 35, 45, 64, 70, 80, 90, 100]
                win_sizes3 = [9, 15, 17, 20, 32, 48, 51, 64]
                win_sizes4 = [6, 7, 9, 11, 13, 15, 19, 26, 30, 35]
                win_sizes5 = [6, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
                win_sizes6 = list(range(5, 16, 2))
            elif query.shape[1] == 30:
                win_sizes1 = [16, 19, 24, 27, 29, 31, 37, 39, 40, 41, 50, 52, 54, 58, 63, 70, 85, 100, 125, 150, 169]  # 修改模板窗口大小
                win_sizes2 = [13, 19, 21, 23, 25, 27, 29, 31, 32, 33, 35, 36, 40, 42, 44, 54, 56, 57, 59, 60, 64, 70, 80, 90, 100]
                win_sizes3 = [8, 10, 13, 15, 17, 19, 21, 22, 23, 24, 26, 28, 31, 35, 38, 40, 42, 44, 46, 48, 50, 51, 60, 64]
                win_sizes4 = [6, 8, 10, 11, 13, 14, 16, 19, 21, 23, 25, 26, 27, 28, 29, 30, 35]
                win_sizes5 = [6, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
                win_sizes6 = list(range(5, 16, 2))
            y1 = attention_map[:, :, :, -edge:].cpu()
            y_mean1 = y1.mean(dim=0)
            grid_list = []
            for i in range(layer):
                x_cor, y_cor = torch.meshgrid(torch.arange(token_scale[layer]), torch.arange(token_scale[layer-i]))
                grid_list.append((x_cor, y_cor))
            x_cor1,y_cor1 = grid_list[0]
            x_cor2,y_cor2 = grid_list[1]
            x_cor3,y_cor3 = grid_list[2]
            x_cor4,y_cor4 = grid_list[3]
            x_cor5,y_cor5 = grid_list[4]
            x_cor6,y_cor6 = grid_list[5]
            cal_rate_1 = []
            window_choose_1 = []
            cal_rate_2 = []
            window_choose_2 = []
            cal_rate_3 = []
            window_choose_3 = []
            cal_rate_4 = []
            window_choose_4 = []
            cal_rate_5 = []
            window_choose_5 = []
            cal_rate_6 = []
            window_choose_6 = []

            mask_save_1 = []
            mask_save_2 = []
            mask_save_3 = []
            mask_save_4 = []
            mask_save_5 = []
            mask_save_6 = []

            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            y4 = attention_map[:, :, :, left_edge4:right_edge4].cpu()
            y5 = attention_map[:, :, :, left_edge5:right_edge5].cpu()
            y6 = attention_map[:, :, :, left_edge6:right_edge6].cpu()
            for i in range(y_mean1.shape[0]):
                # print(f'BLOCK {block_num}')
                # print(f'HEAD {i}')
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                window_3 = attention_map[:, i, :, left_edge3:right_edge3]
                window_4 = attention_map[:, i, :, left_edge4:right_edge4]
                window_5 = attention_map[:, i, :, left_edge5:right_edge5]
                window_6 = attention_map[:, i, :, left_edge6:right_edge6]
                # print(f'last 1 mask = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 2 mask = {y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 3 mask = {y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 4 mask = {y4[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 5 mask = {y5[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                # print(f'last 6 mask = {y6[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_6_1: # 目前选为
                    # print(f'head = {i} mask 1 is skip')
                    # print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    mask_now = torch.zeros_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                    mask_save_1.append(mask_now)
                else:
                    # print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) < win_size
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_deep:
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:]
                            mask_now = mask1.unsqueeze(0)*1.0
                            cal1 = mask1.sum()
                            cal_rate_1.append(int(cal1))
                            window_choose_1.append(win_size)
                            mask_save_1.append(mask_now)
                            mask_tru1 += 1
                            break
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        mask_now = torch.ones_like(y1[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(edge))
                        mask_save_1.append(mask_now)
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_6_2: # 目前选为
                    # print(f'head = {i} mask 2 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    mask_now = torch.zeros_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                    mask_save_2.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    '''第二对角'''
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) < win_size
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                            mask_now = mask2.unsqueeze(0)*1.0
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_save_2.append(mask_now)
                            # print(f'HEAD {i} MASK 2 = {win_size}')
                            mask_tru2 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        mask_now = torch.ones_like(y2[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        mask_save_2.append(mask_now)
                        # print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_6_3: # 目前选为0.1
                    # print(f'head = {i} mask 3 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn3 = torch.zeros_like(y3[:,i,:,:])
                    mask_now = torch.zeros_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_3.append(int(0))
                    window_choose_3.append(int(0))
                    mask_save_3.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    '''第三对角'''
                    # 倒数第三个mask
                    mask_tru3 = 0
                    for win_size in win_sizes3:
                        mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) < win_size
                        # attn_fraction = (mask3 * y_mean3[i]).sum() / y_mean3[i].sum()
                        attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                            mask_now = mask3.unsqueeze(0)*1.0
                            # print(f'HEAD {i} MASK 3 = {win_size}')
                            cal3 = mask3.sum()
                            cal_rate_3.append(int(cal3))
                            window_choose_3.append(win_size)
                            mask_save_3.append(mask_now)
                            mask_tru3 += 1
                            break
                    if mask_tru3 == 0:
                        new_attn3 = y3[:,i,:,:]
                        mask_now = torch.ones_like(y3[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_3.append(cal_original_3)
                        window_choose_3.append(int(right_edge3-left_edge3))
                        mask_save_3.append(mask_now)
                        # print(f'NO USE ANY MASK 3 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge3:right_edge3] = new_attn3

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y4[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_6_4: # 目前选为0.1
                    # print(f'head = {i} mask 4 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge4:right_edge4].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn4 = torch.zeros_like(y4[:,i,:,:])
                    mask_now = torch.zeros_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_4.append(int(0))
                    window_choose_4.append(int(0))
                    mask_save_4.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge4:right_edge4].sum()/attention_map[:,i,:,:].sum()}')
                    '''第四对角'''
                    # 倒数第四个mask
                    mask_tru4 = 0
                    for win_size in win_sizes4:
                        mask4 = torch.abs(x_cor4*(right_edge4-left_edge4)/edge - y_cor4) < win_size
                        # attn_fraction = (mask4 * y_mean4[i]).sum() / y_mean4[i].sum()
                        attn_fraction = (mask4.unsqueeze(0) * y4[:,i,:,:]).sum() / y4[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn4 = mask4.unsqueeze(0) * y4[:,i,:,:] #
                            mask_now = mask4.unsqueeze(0)*1.0
                            # print(f'HEAD {i} MASK 4 = {win_size}')
                            cal4 = mask4.sum()
                            cal_rate_4.append(int(cal4))
                            window_choose_4.append(win_size)
                            mask_save_4.append(mask_now)
                            mask_tru4 += 1
                            break
                    if mask_tru4 == 0:
                        new_attn4 = y4[:,i,:,:]
                        mask_now = torch.ones_like(y4[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_4.append(cal_original_4)
                        window_choose_4.append(int(right_edge4-left_edge4))
                        mask_save_4.append(mask_now)
                        # print(f'NO USE ANY MASK 4 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge4:right_edge4] = new_attn4

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y5[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_6_5: # 目前选为
                    # print(f'head = {i} mask 5 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge5:right_edge5].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn5 = torch.zeros_like(y5[:,i,:,:])
                    mask_now = torch.zeros_like(y5[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_5.append(int(0))
                    window_choose_5.append(int(0))
                    mask_save_5.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge5:right_edge5].sum()/attention_map[:,i,:,:].sum()}')
                    '''第五对角'''
                    # 倒数第五个mask
                    mask_tru5 = 0
                    for win_size in win_sizes5:
                        mask5 = torch.abs(x_cor5*(right_edge5-left_edge5)/edge - y_cor5) < win_size
                        attn_fraction = (mask5.unsqueeze(0) * y5[:,i,:,:]).sum() / y5[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn5 = mask5.unsqueeze(0) * y5[:,i,:,:]
                            mask_now = mask5.unsqueeze(0)*1.0
                            cal5 = mask5.sum()
                            cal_rate_5.append(int(cal5))
                            window_choose_5.append(win_size)
                            mask_save_5.append(mask_now)
                            mask_tru5 += 1
                            break
                    if mask_tru5 == 0:
                        new_attn5 = y5[:,i,:,:]
                        mask_now = torch.ones_like(y5[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_5.append(cal_original_5)
                        window_choose_5.append(int(right_edge5-left_edge5))
                        mask_save_5.append(mask_now)
                attn_aftermask[:, i, :, left_edge5:right_edge5] = new_attn5

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y6[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_6_6: # 目前选为
                    # print(f'head = {i} mask 6 is skip')
                    # print(f'NO bizhi = {attention_map[:, i, :, left_edge6:right_edge6].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn6 = torch.zeros_like(y6[:,i,:,:])
                    mask_now = torch.zeros_like(y6[:,i,:,:])[0].unsqueeze(0)*1.0
                    cal_rate_6.append(int(0))
                    window_choose_6.append(int(0))
                    mask_save_6.append(mask_now)
                else: 
                    # print(f'YES bizhi = {attention_map[:, i, :, left_edge6:right_edge6].sum()/attention_map[:,i,:,:].sum()}')
                    '''第六对角'''
                    # 倒数第六个mask
                    mask_tru6 = 0
                    for win_size in win_sizes6:
                        mask6 = torch.abs(x_cor6*(right_edge6-left_edge6)/edge - y_cor6) < win_size
                        attn_fraction = (mask6.unsqueeze(0) * y6[:,i,:,:]).sum() / y6[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn6 = mask6.unsqueeze(0) * y6[:,i,:,:]
                            mask_now = mask6.unsqueeze(0)*1.0
                            cal6 = mask6.sum()
                            cal_rate_6.append(int(cal6))
                            window_choose_6.append(win_size)
                            mask_save_6.append(mask_now)
                            mask_tru6 += 1
                            break
                    if mask_tru6 == 0:
                        new_attn6 = y6[:,i,:,:]
                        mask_now = torch.ones_like(y6[:,i,:,:])[0].unsqueeze(0)*1.0
                        cal_rate_6.append(cal_original_6)
                        window_choose_6.append(int(right_edge6-left_edge6))
                        mask_save_6.append(mask_now)
                attn_aftermask[:, i, :, left_edge6:right_edge6] = new_attn6

            total_sum_oneblock_1 = sum([int(x) for x in cal_rate_1])
            total_sum_oneblock_2 = sum([int(x) for x in cal_rate_2])
            total_sum_oneblock_3 = sum([int(x) for x in cal_rate_3])
            total_sum_oneblock_4 = sum([int(x) for x in cal_rate_4])
            total_sum_oneblock_5 = sum([int(x) for x in cal_rate_5])
            total_sum_oneblock_6 = sum([int(x) for x in cal_rate_6])

            block_cal_token_num_list_1[block_num] = total_sum_oneblock_1
            block_cal_token_num_list_2[block_num] = total_sum_oneblock_2
            block_cal_token_num_list_3[block_num] = total_sum_oneblock_3
            block_cal_token_num_list_4[block_num] = total_sum_oneblock_4
            block_cal_token_num_list_5[block_num] = total_sum_oneblock_5
            block_cal_token_num_list_6[block_num] = total_sum_oneblock_6

            window_wid_chosen_1[block_num] = window_choose_1
            window_wid_chosen_2[block_num] = window_choose_2
            window_wid_chosen_3[block_num] = window_choose_3
            window_wid_chosen_4[block_num] = window_choose_4
            window_wid_chosen_5[block_num] = window_choose_5
            window_wid_chosen_6[block_num] = window_choose_6

            mask_chosen_save_1[block_num] = mask_save_1
            mask_chosen_save_2[block_num] = mask_save_2
            mask_chosen_save_3[block_num] = mask_save_3
            mask_chosen_save_4[block_num] = mask_save_4
            mask_chosen_save_5[block_num] = mask_save_5
            mask_chosen_save_6[block_num] = mask_save_6

            if block_num == model_depth-1:
                # 第一个对角
                # print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                # print(f'Mask we save: {mask_chosen_save_1}')
                file_name = f'mask_17.pth'
                torch.save(mask_chosen_save_1, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                # print(f'Total calculation: {cal_token_aftermask_1}')
                # print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                
                # 第二个对角
                # print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                # print(f'Mask we save: {mask_chosen_save_2}')
                file_name = f'mask_18.pth'
                torch.save(mask_chosen_save_2, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                # print(f'Total calculation: {cal_token_aftermask_2}')
                # print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                
                # 第三个对角
                # print('NOW FOR MASK 3    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_3}')
                # print(f'Mask we save: {mask_chosen_save_3}')
                file_name = f'mask_19.pth'
                torch.save(mask_chosen_save_3, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_3}')
                cal_token_aftermask_3 = sum(block_cal_token_num_list_3.values())
                # print(f'Total calculation: {cal_token_aftermask_3}')
                # print(f'Without mask: {cal_original_3*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_3/(cal_original_3*model_depth*model_depth)}')
                
                # 第四个对角
                # print('NOW FOR MASK 4    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_4}')
                # print(f'Mask we save: {mask_chosen_save_4}')
                file_name = f'mask_20.pth'
                torch.save(mask_chosen_save_4, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_4}')
                cal_token_aftermask_4 = sum(block_cal_token_num_list_4.values())
                # print(f'Total calculation: {cal_token_aftermask_4}')
                # print(f'Without mask: {cal_original_4*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_4/(cal_original_4*model_depth*model_depth)}')

                # 第五个对角
                # print('NOW FOR MASK 5    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_5}')
                # print(f'Mask we save: {mask_chosen_save_5}')
                file_name = f'mask_21.pth'
                torch.save(mask_chosen_save_5, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_5}')
                cal_token_aftermask_5 = sum(block_cal_token_num_list_5.values())
                # print(f'Total calculation: {cal_token_aftermask_5}')
                # print(f'Without mask: {cal_original_5*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_5/(cal_original_5*model_depth*model_depth)}')

                # 第六个对角
                # print('NOW FOR MASK 6    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_6}')
                # print(f'Mask we save: {mask_chosen_save_6}')
                file_name = f'mask_22.pth'
                torch.save(mask_chosen_save_6, os.path.join(root_path, file_name)) # 保存 Mask we save
                # print(f'block_cal_token_num_list: {block_cal_token_num_list_6}')
                cal_token_aftermask_6 = sum(block_cal_token_num_list_6.values())
                # print(f'Total calculation: {cal_token_aftermask_6}')
                # print(f'Without mask: {cal_original_6*model_depth*model_depth}')
                # print(f'Calculation rate: {cal_token_aftermask_6/(cal_original_6*model_depth*model_depth)}')
                
                ###################
                # 打印全图总计数
                before = left_edge6*edge
                image_after_tokens = (cal_token_aftermask_1+cal_token_aftermask_2+cal_token_aftermask_3+\
                                      cal_token_aftermask_4+cal_token_aftermask_5+cal_token_aftermask_6)+before*model_depth*model_depth
                image_before_tokens = (cal_original_1+cal_original_2+cal_original_3+\
                                       cal_original_4+cal_original_5+cal_original_6+before)*model_depth*model_depth
                print(f'Image calculation after mask: {image_after_tokens}')
                print(f'Image calculation before mask: {image_before_tokens}')
                print(f'Image calculation rate: {(image_before_tokens-image_after_tokens)/image_before_tokens}')

                # 清除字典
                dict_list = [block_cal_token_num_list_1, block_cal_token_num_list_2, block_cal_token_num_list_3,\
                             block_cal_token_num_list_4, block_cal_token_num_list_5, block_cal_token_num_list_6]
                window_list = [window_wid_chosen_1, window_wid_chosen_2, window_wid_chosen_3,\
                               window_wid_chosen_4, window_wid_chosen_5, window_wid_chosen_6]

                for d, w in zip(dict_list, window_list):
                    d.clear()
                    w.clear()
                    # print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    # print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')

            out = attn_aftermask @ value

    return out