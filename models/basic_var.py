import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.quantize_all_me import WeightPCQuantizer, WeightPGQuantizer, WeightPCQuantileQuantizer, WeightPGQuantileQuantizer
# from models.quantize_all_me import BaseQuantizer, ActDynamicQuantizer, ActPGDynamicQuantizer
# from global_vars import MODEL_DEPTH, MODE

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

from models.helpers import DropPath, drop_path
# from models.caogao import slow_attn_diff, slow_attn_diff2, slow_attn_test
# from models.dididi import slow_attn_cal_dis
from models.mask_attn import slow_attn_after_mask
from models.mask_attn_try import slow_attn_after_mask_try
from models.mask_for_inference import mask_inference, quant_mask_inference
# from models.inference_cfg import mask_inference_cfg_global
# from models.test import slow_attn_ban_mini_token
from models.ban_mini_token import slow_attn_ban_mini_token_2
from models.only_observe_matrix import observe
from models.quant_self_attention import QuantLinear, QuantSelfAttention
# this file only provides the 3 blocks used in VAR transformer
__all__ = ['FFN', 'AdaLNSelfAttn', 'AdaLNBeforeHead']

# automatically import fused operators
dropout_add_layer_norm = fused_mlp_func = memory_efficient_attention = flash_attn_func = None


# try:
#     from flash_attn.ops.layer_norm import dropout_add_layer_norm
#     from flash_attn.ops.fused_dense import fused_mlp_func
#     print("WE USE FLASH VERSION###############################################################")
# except ImportError: pass
# # automatically import faster attention implementations
# try: from xformers.ops import memory_efficient_attention
# except ImportError: pass
# try: from flash_attn import flash_attn_func              # qkv: BLHc, ret: BLHcq
# # try: from flash_attention_me import flash_attn_func        # 我的kqv
# except ImportError: pass
# try: from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
# except ImportError: pass
# except ImportError:
#     def slow_attn(query, key, value, scale: float, attn_mask=None, dropout_p=0.0):
#         attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
#         if attn_mask is not None: attn.add_(attn_mask)
#         attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
#         # print(f"si can be print as {si}")
#         # torch.save(attention_map,f"./attentionmap{si}.pth")
#      
#    return (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1)) @ value
            # quant_query=self.query_quantizer(q)
            # quant_key=self.key_quantizer(k)
            # attn_mask=attn_bias
            # quant_attn = quant_query.mul(raw_layer.scale) @ quant_key.transpose(-2, -1) # BHLc @ BHcL => BHLL
            # if attn_mask is not None: 
            #     quant_attn.add_(attn_mask)
            # attention_map = (F.dropout(quant_attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else quant_attn.softmax(dim=-1))
            
            # quant_attention_map=self.attn_map_quantizer(attention_map)
            # quant_value=self.value_quantizer(attention_map)
            # quant_oup=(quant_attention_map @ quant_value).transpose(1, 2).reshape(B, L, C)
            # oup = quant_oup
            # attnmap = quant_cal_attnmap(query=quant_query, key=quant_key, scale=raw_layer.scale, attn_mask=attn_bias, dropout_p=dropout_p)
            

def quant_cal_attnmap(query, key, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: attn.add_(attn_mask)
    attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
    return attention_map

def slow_attn_original(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: attn.add_(attn_mask)
    attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
    return attention_map @ value

def distance_to_line(x, y, a, b, c):
    return abs(a*x + b*y + c)/ math.sqrt(a**2 + b**2)

def slow_attn(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
    print(f'block_num={block_num}')
    level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
    token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
    tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
    for index, this_square in enumerate(token_scale):
        if query.shape[2] == this_square:
            layer = index
            break
    # root_path = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/nosoftmax/16/scale{layer}/'
    # os.makedirs(root_path, exist_ok=True)
    # root_path = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/perblock/16/scale{layer}/'
    # os.makedirs(root_path, exist_ok=True)
    attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
    if attn_mask is not None: attn.add_(attn_mask)
    attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
    file_name = f'block_{block_num}.pth'
    # if layer > 2:
    #     attention_map[:, 4, :, 5:] = 0.0
    if layer > 2:
        attention_map[:, 10, :, 5:] = 0.0
    if layer > 5:
        attention_map[:, 28, :, 1:] = 0.0
    # file_name = f'scale{layer}.pth'
    # torch.save(attn, os.path.join(root_path, file_name)) # 保存不softmax的attn
    
    # file_name = f"block_{file_idx}.pth"
    
    
    # while os.path.exists(os.path.join(root_path, file_name)):
    #     file_idx += 1
    #     file_name = f"block_{file_idx}.pth"
    # torch.save(attn, os.path.join(root_path, file_name))

    # torch.save(attention_map, root_path + f"attentionmap0.pth")
     
    return attention_map @ value
 

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2) # 
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class FFN(nn.Module):
    def __init__(self, block_idx, in_features, hidden_features=None, out_features=None, drop=0., fused_if_available=True):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_if_available else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
        self.block_idx = block_idx
        self.scale_idx = -1

    def forward(self, x):
        # breakpoint()
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x, weight1=self.fc1.weight, weight2=self.fc2.weight, bias1=self.fc1.bias, bias2=self.fc2.bias,
                activation='gelu_approx', save_pre_act=self.training, return_residual=False, checkpoint_lvl=0,
                heuristic=0, process_group=None,
            ))
        else:
            x = self.fc1(x)
            x = self.act(x)
            '''打一下fc2的channel,查看是否存在channel不均衡问题'''
            # # breakpoint()
            
            # if self.block_idx == 0:
            #     self.scale_idx += 1
            #     print(f"\nNOW SCALE {self.scale_idx}")

            # max_range, min_range, channel_info = print_sth_about_channel(x)
            # # 打印结果
            # print(f"block {self.block_idx} fc2_in ———— Min-Max range across channels:{max_range-min_range}")
            # print(f"max act : min act = {channel_info.T}")
            # if self.scale_idx == 9 and self.block_idx == 29:
            #     print(f"\n NEXT PICTURE \n")
            #     self.scale_idx = -1
        
            x = self.fc2(x)

            x = self.drop(x)
            return x
            # return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp_func={self.fused_mlp_func is not None}'

class SelfAttention(nn.Module):
    def __init__(
        self, block_idx, embed_dim=768, num_heads=12,
        attn_drop=0., proj_drop=0., attn_l2_norm=False, flash_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        # breakpoint()
        self.block_idx, self.num_heads, self.head_dim = block_idx, num_heads, embed_dim // num_heads  # 0,30,64(1920/30)
        self.attn_l2_norm = attn_l2_norm # True
        if self.attn_l2_norm:
            self.scale = 1
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)# 1 30 1 1
            self.max_scale_mul = torch.log(torch.tensor(100)).item() 
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False) # 1920 1920*3
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim)) # [1920]
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim) # 1920 1920
        self.proj_drop = nn.Dropout(proj_drop, inplace=True) if proj_drop > 0 else nn.Identity()
        self.attn_drop: float = attn_drop # 0.0
        self.using_flash = flash_if_available and flash_attn_func is not None
        self.using_xform = flash_if_available and memory_efficient_attention is not None
        
        # only used during inference
        self.caching, self.cached_k, self.cached_v = False, None, None
    
    def kv_caching(self, enable: bool): 
        self.caching, self.cached_k, self.cached_v = enable, None, None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, attn_bias):
        # using_flash=False
        # self.using_xform=False
        # breakpoint()
        
        B, L, C = x.shape # 16 1 1920
        if isinstance(self.mat_qkv, nn.Linear):
            assert self.mat_qkv.bias is None
        elif isinstance(self.mat_qkv, QuantLinear):
            assert self.mat_qkv.raw_layer.bias is None
        pre_qkv=self.mat_qkv(x) # [16, 1, 5760]
        bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
        pre_qkv+=bias
        qkv=pre_qkv.view(B, L, 3, self.num_heads, self.head_dim) # [16, 1, 3, 30, 64]

        # qkv = F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))).view(B, L, 3, self.num_heads, self.head_dim)
        main_type = qkv.dtype
        # qkv: BL3Hc
        
        using_flash = self.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or self.using_xform: q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            if using_flash or self.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        # print(f'q.shape={q.shape}')
        # print(f'k.shape={k.shape}')
        # print(f'v.shape={v.shape}')

        if self.caching:
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=dim_cat); v = self.cached_v = torch.cat((self.cached_v, v), dim=dim_cat)
        # 这里把k和v的shape改了，q却没变，与后面说qk都是BHLc对不上，strange#################################

        dropout_p = self.attn_drop if self.training else 0.0
        
        if using_flash:
            oup = flash_attn_func(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), dropout_p=dropout_p, softmax_scale=self.scale).view(B, L, C)
        elif self.using_xform:
            oup = memory_efficient_attention(q.to(dtype=main_type), k.to(dtype=main_type), v.to(dtype=main_type), attn_bias=None if attn_bias is None else attn_bias.to(dtype=main_type).expand(B, self.num_heads, -1, -1), p=dropout_p, scale=self.scale).view(B, L, C)
        else:
            block_num = self.block_idx
            block_cal_token_num_list_1 = {}  # 创建一个空的列表来存储每个 block 的计数值和 mask后的tokenvs原本toke计数
            # oup = slow_attn(query=q, key=k, value=v, block_num=block_num, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            # oup = slow_attn_after_mask_try(query=q, key=k, value=v, block_num=block_num, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            if MODE == -1:
                # print(f'MODE = -1')
                oup = slow_attn_ban_mini_token_2(query=q, key=k, value=v, block_num=block_num, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            # oup = observe(query=q, key=k, value=v, block_num=block_num, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            
            '''inference use below'''
            if MODE == 0:
                # breakpoint()
                oup = slow_attn_original(query=q, key=k, value=v, block_num=block_num, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
                # def slow_attn_original(query, key, value, block_num, scale: float, attn_mask=None, dropout_p=0.0):
                #     attn = query.mul(scale) @ key.transpose(-2, -1) # BHLc @ BHcL => BHLL
                #     if attn_mask is not None: attn.add_(attn_mask)
                #     attention_map = (F.dropout(attn.softmax(dim=-1), p=dropout_p, inplace=True) if dropout_p > 0 else attn.softmax(dim=-1))
                #     return attention_map @ value
            elif MODE == 1 or MODE == 2:
                # print(f'MODE = 1 OR 2')
                oup = mask_inference(query=q, key=k, value=v, 
                                     block_num=block_num, 
                                    #  cuda = CUDA_DEVICES, 
                                    #  model_depth = MODEL_DEPTH, 
                                    #  mode = MODE,
                                    #  pic_num = PIC_NUM,
                                    #  seed = SEED,
                                     scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            # elif MODE == 2:
            #     oup = mask_inference_cfg_global(query=q, key=k, value=v, block_num=block_num, scale=self.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
            
            # 画attention output(就是zhihang paper中橙色那个)
            # breakpoint()
            qkv_map = self.proj(oup)
            # print(f'qkv_map.shape={qkv_map.shape}')
            # torch.save(qkv_map, "/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/qkv_16/qkv.pth")
        return self.proj_drop(self.proj(oup))
        # attn = (q @ k.transpose(-2, -1)).add_(attn_bias + self.local_rpb())  # BHLc @ BHcL => BHLL
        # attn = self.attn_drop(attn.softmax(dim=-1))
        # oup = (attn @ v).transpose_(1, 2).reshape(B, L, -1)     # BHLL @ BHLc = BHLc => BLHc => BLC
    
    def extra_repr(self) -> str:
        return f'using_flash={self.using_flash}, using_xform={self.using_xform}, attn_l2_norm={self.attn_l2_norm}'
    
class AdaLNSelfAttn(nn.Module):
    def __init__(
        self, block_idx, last_drop_p, embed_dim, cond_dim, shared_aln: bool, norm_layer,
        num_heads, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., attn_l2_norm=False,
        flash_if_available=False, fused_if_available=True,
    ):
        super(AdaLNSelfAttn, self).__init__()
        self.block_idx, self.last_drop_p, self.C = block_idx, last_drop_p, embed_dim
        self.C, self.D = embed_dim, cond_dim
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.attn = SelfAttention(block_idx=block_idx, embed_dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, attn_l2_norm=attn_l2_norm, flash_if_available=flash_if_available)
        self.ffn = FFN(block_idx=block_idx, in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio), drop=drop, fused_if_available=fused_if_available)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin)
        
        self.fused_add_norm_fn = None
    
    # NOTE: attn_bias is None during inference because kv cache is enabled
    def forward(self, x, cond_BD, block_idx, attn_bias):   # C: embed_dim, D: cond_dim
        if self.shared_aln:
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
        else:
            
            gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        # x = x + self.drop_path(self.attn( self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
        # breakpoint()
        
        ''''''
        # # 将张量reshape成[1920, 16*n]
        # # breakpoint()
        # # x:[16, 1, 1920] reshaped_tensor:[1920, 16]
        # # reshaped_tensor = x.reshape(-1,1920).transpose(0,1)
        # print(f'channel num = {x.shape[-1]}')
        # reshaped_tensor = x.reshape(-1,x.shape[-1].transpose(0,1))
        # # 计算每个channel的最大值和最小值
        # channel_max = torch.max(reshaped_tensor, dim=1)[0] # or torch.max(reshaped_tensor, dim=1).values # 0:values 1:indices
        # channel_min = torch.min(reshaped_tensor, dim=1)[0]
    
        # # 计算每个通道的range（最大值-最小值）
        # channel_range = channel_max - channel_min # [1920] 每个通道的range
        # max_range = torch.max(channel_range) # tensor(0.2281, device='cuda:0')
        # min_range = torch.min(channel_range) # tensor(0.0062, device='cuda:0')
        # breakpoint()
        # 检查层名称是否包含 "fc2"
        
        # max_range, min_range = print_sth_about_channel(x)
        # # 打印结果
        # print(f"layer {self.__class__.__name__} attn_in ———— Min-Max range across channels:{max_range-min_range}")
        
        # '''attn'''
        x = x + self.drop_path(self.attn(self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1), attn_bias=attn_bias).mul_(gamma1))
        # # 检查层名称是否包含 "fc2"
        
        # max_range, min_range = print_sth_about_channel(x)
        # # 打印结果
        # print(f"layer {self.__class__.__name__} attn_out/ffn_in ———— Min-Max range across channels:{max_range-min_range}")

        '''ffn'''
        x = x + self.drop_path(self.ffn( self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed when FusedMLP is used
        
        # # 检查层名称是否包含 "fc2"
        

        # max_range, min_range = print_sth_about_channel(x)
        # # 打印结果
        # print(f"layer {self.__class__.__name__} ffn_out ———— Min-Max range across channels:{max_range-min_range}")

        ''''''
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, norm_layer):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), nn.Linear(D, 2*C))
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: torch.Tensor):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)


def print_sth_about_channel(x):
    print(f'channel num = {x.shape[-1]}')
    reshaped_tensor = x.reshape(-1,x.shape[-1]).transpose(0,1)
    # 计算每个channel的最大值和最小值
    channel_max = torch.max(reshaped_tensor, dim=1)[0] # or torch.max(reshaped_tensor, dim=1).values # 0:values 1:indices
    channel_min = torch.min(reshaped_tensor, dim=1)[0]
    # 创建一个用于存储索引的张量
    indices = torch.arange(channel_max.size(0)).cuda()
    # breakpoint()
    # 使用 torch.stack 将索引和值堆叠起来，并在新的维度上创建一个张量
    channel_info = torch.stack((indices, channel_max, channel_min), dim=1)
    # 计算每个通道的range（最大值-最小值）
    channel_range = channel_max - channel_min # [1920] 每个通道的range
    max_range = torch.max(channel_range) # tensor(0.2281, device='cuda:0')
    min_range = torch.min(channel_range) # tensor(0.0062, device='cuda:0')
    return max_range.item(), min_range.item(), channel_info
    # # 打印结果
    # print(f"attn_in Min-Max range across channels:{max_range.item()-min_range.item()}")