# 测一下scale5
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
threshold_deep = 0.95
mini_threshold_2 = 0.2
mini_threshold_3 = 0.15
mini_threshold_4 = 0.15
mini_threshold_5 = 0.2
mini_threshold_6 = 0.2
mini_threshold_7 = 0.2
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
def observe(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    root_path = '/share/xierui-nfs/pythonProgram/VAR/temp/'
    os.makedirs(root_path, exist_ok=True)
    model_depth = query.shape[1]
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
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
        # attention_map = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention16/attentionmap{layer}.pth')
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
        if layer != 5:
            print(f'##################    BLOCK {block_num}    ######################')
            print(f'##################    SCALE {layer}    ######################')
            out = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
    
        '''三个mask的情况:layer=4和layer=5'''
        if layer == 5:
            print(f'##################    BLOCK {block_num}    ######################')
            print(f'##################    SCALE {layer}    ######################')
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
            '''先排除特别小的点 整个attention_map小于0.003的点全部置零'''
            # '''更改：如果首两列加起来<0.1就先不抹了，这可能是个整张图都很小的情况'''
            mask = torch.lt(attention_map[:, :, :, :], 0.003)
            attention_map[:, :, :, :].masked_fill_(mask, 0)
            y1 = attention_map[:, :, :, -edge:].cpu() # 排除极小点之后的y1
            y2 = attention_map[:, :, :, left_edge2:right_edge2].cpu()
            y3 = attention_map[:, :, :, left_edge3:right_edge3].cpu()
            for i in range(y_mean1.shape[0]):
                print(f'HEAD {i}')
                window_2 = attention_map[:, i, :, left_edge2:right_edge2]
                window_3 = attention_map[:, i, :, left_edge3:right_edge3]

                ''''''''''''
                print(f'last 2 mask = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                print(f'last 3 mask = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                
                # 创建一个二维张量
                for b in range(16):
                    tensor = attention_map[b,i]
                    print(tensor.shape)

                    # 查找最大值及其对应的位置坐标
                    max_value, max_indices = torch.max(tensor, dim=1)

                    # 打印最大值
                    print(f'B={b}的最大值为:{max_value}')

                    # 打印最大值对应的位置坐标
                    
                    print(f'B={b}的最大值的位置坐标:{max_indices}')
                one_tensor = attention_map[0,i]
                # 遍历每一行
                for j in range(one_tensor.size(0)):
                    # 求取当前行的和
                    row_sum = torch.sum(tensor[j])
                    
                    # 打印当前行的和以及行索引
                    print(f"B=0 的第 {j+1} 行的和: {row_sum}")
                ''''''''''''
                

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_3: # 目前选为0.2
                    print(f'head = {i} mask 1 is skip')
                    print(f'NO bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                    
                    
                    new_attn1 = torch.zeros_like(y1[:,i,:,:])
                    cal_rate_1.append(int(0))
                    window_choose_1.append(int(0))
                else:
                    print(f'YES bizhi = {y1[:,i,:,:].sum()/attention_map[:,i,:,:].sum()}')
                
                    # 倒数第一个mask
                    mask_tru1 = 0
                    for win_size in win_sizes1:
                        mask1 = torch.abs(x_cor1 - y_cor1) <= win_size # ([16, 16]),后续诸如([169, 169])
                        # attn_fraction = (mask1 * y_mean1[i]).sum() / y_mean1[i].sum()
                        attn_fraction = (mask1.unsqueeze(0) * y1[:,i,:,:]).sum() / y1[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            # new_attn1 = mask1 * y1
                            new_attn1 = mask1.unsqueeze(0) * y1[:,i,:,:] # ([1, 169, 169])*([16, 169, 169])=([16, 169, 169])
                            # print(f'HEAD {i} MASK 1 = {win_size}')
                            cal1 = mask1.sum()
                            cal_rate_1.append(int(cal1))
                            window_choose_1.append(win_size)
                            mask_tru1 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru1 == 0:
                        new_attn1 = y1[:,i,:,:]
                        cal_rate_1.append(cal_original_1)
                        window_choose_1.append(int(edge))
                        # print(f'NO USE ANY MASK 1 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask = attention_map
                attn_aftermask[:, i, :, -edge:] = new_attn1

                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y2[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_3: # 目前选为0.2
                    print(f'head = {i} mask 2 is skip')
                    print(f'NO bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn2 = torch.zeros_like(y2[:,i,:,:])
                    cal_rate_2.append(int(0))
                    window_choose_2.append(int(0))
                else: 
                    print(f'YES bizhi = {attention_map[:, i, :, left_edge2:right_edge2].sum()/attention_map[:,i,:,:].sum()}')
                    '''如果存在第二对角'''
                    # 倒数第二个mask
                    mask_tru2 = 0
                    for win_size in win_sizes2:
                        mask2 = torch.abs(x_cor2*(right_edge2-left_edge2)/edge - y_cor2) <= win_size
                        # attn_fraction = (mask2 * y_mean2[i]).sum() / y_mean2[i].sum()
                        attn_fraction = (mask2.unsqueeze(0) * y2[:,i,:,:]).sum() / y2[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            # new_attn1 = mask1 * y1
                            new_attn2 = mask2.unsqueeze(0) * y2[:,i,:,:] #
                            # print(f'HEAD {i} MASK 2 = {win_size}')
                            cal2 = mask2.sum()
                            cal_rate_2.append(int(cal2))
                            window_choose_2.append(win_size)
                            mask_tru2 += 1
                            break
                        # else:
                        #     print(f'win_size {win_size} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                    if mask_tru2 == 0:
                        new_attn2 = y2[:,i,:,:]
                        cal_rate_2.append(cal_original_2)
                        window_choose_2.append(int(right_edge2-left_edge2))
                        # print(f'NO USE ANY MASK 2 IN HEAD {i}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                attn_aftermask[:, i, :, left_edge2:right_edge2] = new_attn2
                '''排除一个mask矩形内的token值占总比很小的情况(即attnmap某一区域很黑的情况)'''
                if y3[:,i,:,:].sum()/attention_map[:,i,:,:].sum() < mini_threshold_3: # 目前选为0.2
                    print(f'head = {i} mask 3 is skip')
                    print(f'NO bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    
                    new_attn3 = torch.zeros_like(y3[:,i,:,:])
                    cal_rate_3.append(int(0))
                    window_choose_3.append(int(0))
                else: 
                    print(f'YES bizhi = {attention_map[:, i, :, left_edge3:right_edge3].sum()/attention_map[:,i,:,:].sum()}')
                    '''如果存在第二对角'''
                    # 倒数第三个mask
                    mask_tru3 = 0
                    for win_size in win_sizes3:
                        mask3 = torch.abs(x_cor3*(right_edge3-left_edge3)/edge - y_cor3) <= win_size
                        # attn_fraction = (mask3 * y_mean3[i]).sum() / y_mean3[i].sum()
                        attn_fraction = (mask3.unsqueeze(0) * y3[:,i,:,:]).sum() / y3[:,i,:,:].sum()
                        if attn_fraction >= threshold_all:
                            new_attn3 = mask3.unsqueeze(0) * y3[:,i,:,:] #
                            # print(f'HEAD {i} MASK 3 = {win_size}')
                            cal3 = mask3.sum()
                            cal_rate_3.append(int(cal3))
                            window_choose_3.append(win_size)
                            mask_tru3 += 1
                            break
                    if mask_tru3 == 0:
                        new_attn3 = y3[:,i,:,:]
                        cal_rate_3.append(cal_original_3)
                        window_choose_3.append(int(right_edge3-left_edge3))
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

            if block_num == model_depth-1:
                # 第一个对角
                print('NOW FOR MASK 1     !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_1}')
                print(f'block_cal_token_num_list: {block_cal_token_num_list_1}')
                cal_token_aftermask_1 = sum(block_cal_token_num_list_1.values())
                print(f'Total calculation: {cal_token_aftermask_1}')
                print(f'Without mask: {cal_original_1*model_depth*model_depth}')
                print(f'Calculation rate: {cal_token_aftermask_1/(cal_original_1*model_depth*model_depth)}')
                
                # 第二个对角
                print('NOW FOR MASK 2    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_2}')
                print(f'block_cal_token_num_list: {block_cal_token_num_list_2}')
                cal_token_aftermask_2 = sum(block_cal_token_num_list_2.values())
                print(f'Total calculation: {cal_token_aftermask_2}')
                print(f'Without mask: {cal_original_2*model_depth*model_depth}')
                print(f'Calculation rate: {cal_token_aftermask_2/(cal_original_2*model_depth*model_depth)}')
                
                # 第三个对角
                print('NOW FOR MASK 3    !!!!!!!!!!!!!!!!!!!!!!!')
                print(f'Window we choose: {window_wid_chosen_3}')
                print(f'block_cal_token_num_list: {block_cal_token_num_list_3}')
                cal_token_aftermask_3 = sum(block_cal_token_num_list_3.values())
                print(f'Total calculation: {cal_token_aftermask_3}')
                print(f'Without mask: {cal_original_3*model_depth*model_depth}')
                print(f'Calculation rate: {cal_token_aftermask_3/(cal_original_3*model_depth*model_depth)}')
                
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
                    print(f'NOW CLEAR THE DIC OF TOKENS: {d}')
                    print(f'NOW CLEAR THE DIC OF WINDOW_WID: {w}')

            out = attn_aftermask @ value

    return out