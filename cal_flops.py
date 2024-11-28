import os
import math

B = 8
model_depth = 30
D = 1920
L = 1+4+9+16+25+36+64+100+169+256
L_for4k = 64*L
L_next = L-1
C = 64
H = 30
E = 32
L_mul_Ls = 0
L_mul_Ls_for4k = 0
token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
token_scale_for4k = [64 * x for x in token_scale]
tokenall_scale_for4k = [64* x for x in tokenall_scale]
attn_token = []
attn_token_for4k = []
for i,i_num in enumerate(token_scale):
    for j,j_num in enumerate(tokenall_scale):
        if j == i:
            attn_token.append(i_num*j_num)
print(attn_token)
for i,i_num in enumerate(token_scale_for4k):
    for j,j_num in enumerate(tokenall_scale_for4k):
        if j == i:
            attn_token_for4k.append(i_num*j_num)        

for i in attn_token:
    L_mul_Ls += i
print(L_mul_Ls)
for j in attn_token_for4k:
    L_mul_Ls_for4k += j

ada_lin_1 = 2*(2*B)*L*D*(6*D)*model_depth
mat_qkv = 2*(2*B)*L*D*(3*D)*model_depth
proj = 2*(2*B)*L*D*(D)*model_depth
fc1 = 2*(2*B)*L*D*(4*D)*model_depth
fc2 = 2*(2*B)*L*D*(4*D)*model_depth

q_mul_k = 2*(2*B)*H*L_mul_Ls*C*model_depth
attn_mul_v = 2*(2*B)*H*L_mul_Ls*C*model_depth

q_mul_k_for4k = 2*(2*B)*H*L_mul_Ls_for4k*C*model_depth
attn_mul_v_for4k = 2*(2*B)*H*L_mul_Ls_for4k*C*model_depth

#####
word_emb = 2*B*E*D*L_next
head = 2*(2*B)*L*D*(4*D)
head_num = 2*(2*B)*L*D*(2*D)
flops_linear = ada_lin_1 + mat_qkv + proj + fc1 + fc2 + word_emb + head + head_num
flops_attn = q_mul_k + attn_mul_v
flops_attn_for4k = q_mul_k_for4k + attn_mul_v_for4k
flops_attn = flops_attn
flops_attn_after_mask = int((1-0.7034)*flops_attn)
flops_attn_after_only_cfg = flops_attn/2
flops_attn_after_mask_and_cfg = flops_attn_after_mask/2
# print(f'flops of linear = {flops_linear}')
print(f'flops of attn calculation = {flops_attn}')
flops_original=flops_linear + flops_attn
flops_original_for4k = flops_linear*64 + flops_attn_for4k
flops_only_mask=flops_linear + flops_attn_after_mask
flops_only_cfg=flops_linear + flops_attn_after_only_cfg
flops_mask_cfg=flops_linear + flops_attn_after_mask_and_cfg
print(f'flops of all original = {flops_original}')
print(f"计算的占比为：{flops_attn/flops_original}")
print(f"4k图像计算的占比为：{flops_attn_for4k/flops_original_for4k}")
print(f'flops of all mask = {flops_only_mask}')
print(f'flops of all mask + cfg = {flops_mask_cfg}')
print(f'flops of all only cfg = {flops_only_cfg}')
print(f'mask save flops = {(flops_original-flops_only_mask)/flops_original}')
print(f'cfg save flops = {(flops_original-flops_only_cfg)/flops_original}')
print(f'mask+cfg save flops = {(flops_original-flops_mask_cfg)/flops_original}')
# B=8
# L=[1, 4, 9, 16, 25, 36, 64, 100, 169, 256]

# print(L)
# D=1920
# E=16
# K=3
# W=H=16
# C=640
# ada_flops=0
# ada_mem=0
# quant_flops=0
# quant_mem=0
# L_base=0
# x_flops=[]
# y_flops=[]
# x_mem=[]
# temp_m=[]
# temp_m3=[]
# y_mem=[]
# num=[]
# num_m=[]
# y_attn=[]
# y_block=[]
# count=0
# attn_flops =0
# for i in range(10):
#     num_m.append(len(y_flops))
#     temp=[24*B*L[i]*D**2,12*B*L[i]*D**2,4*B*L[i]*(L[i]+L_base)*D,4*B*L[i]*(L[i]+L_base)*D,4*B*L[i]*D**2,16*B*L[i]*D**2,16*B*L[i]*D**2]
    
#     temp_w=[6*D*(D+1)/4,3*D*(D+1)/4,2*B*(L[i]+L_base)*D/2,2*B*(L[i]+L_base)*D/2,D*(D+1)/4,4*D*(D+1)/4,D*(4*D+1)]
#     temp_i=[2*B*L[i]*D/2,2*B*L[i]*D/2,2*B*L[i]*D/2,2*B*16*L[i]*(L[i]+L_base)/2,2*B*L[i]*D/2,2*B*L[i]*D/2,2*B*L[i]*4*D]
#     #print(2*B*16*L[i]*(L[i]+L_base))
#     temp_m=[x + y for x, y in zip(temp_w, temp_i)]
#     for j in range(30):
#         attn_flops += 4*B*L[i]*(L[i]+L_base)*D + 4*B*L[i]*(L[i]+L_base)*D
#         y_flops.extend(temp)
#         y_mem.extend(temp_m)
#     y_attn.append(4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2)
#     y_block.append(24*B*L[i]*D**2+12*B*L[i]*D**2+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*(L[i]+L_base)*D+4*B*L[i]*D**2+16*B*L[i]*D**2+16*B*L[i]*D**2)
    

#     temp3=[8*B*L[i]*D**2,16*B*L[i]*D**2]
#     temp_w3=[2*D*(D+1)/4,4*D*(D+1)/4]
#     temp_i3=[2*B*L[i]*D/2,2*B*L[i]*D/2]
#     temp_m3=[x + y for x, y in zip(temp_w3, temp_i3)]
    
#     y_flops.extend(temp3)
#     y_mem.extend(temp_m3)
#     #print(len(y_mem))
#     if i<9:
#         y_flops.append(2*B*L[i+1]*E*D) #word embedding部分
#         #print(y_flops)
#         y_mem.append(B*L[i+1]*E+E*D)
    
#     L_base+=L[i]

# y_mem=[i * 2  for i in y_mem]
# y_mem=[sum(y_mem[:i+1]) for i in range(len(y_mem))]
# print(max(y_mem)/1024/1024/1024)