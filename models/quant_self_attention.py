################## 1. Download checkpoints and build models
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
from models.quantize_all_me import WeightPCQuantizer, WeightPGQuantizer, WeightPCQuantileQuantizer, WeightPGQuantileQuantizer
from models.quantize_all_me import BaseQuantizer, ActDynamicQuantizer, ActPGDynamicQuantizer, ActStaticScaleQuantizer
import torch.nn as nn
import torch.nn.functional as F
from models.mask_for_inference import mask_inference, quant_mask_inference
import argparse
DEBUG=False
# QUANT=2


# 创建解析器
parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--cuda_devices', type=str, default="4,", help='Specify the CUDA devices to use, e.g., "0,1" for using devices 0 and 1')
parser.add_argument('--model_depth', type=int, choices=[16, 20, 24, 30], required=True, help='Specify the model depth')
parser.add_argument('--mode', type=int, choices=[0, 1, 2, -1], required=True, help='Specify the mode: 0 for original, 1 for masked, 2 for global cfg, -1 for design mask')
parser.add_argument('--pic_num', type=int, default=250, help='Specify the number of images to generate')
parser.add_argument('--seed', type=int, default=0, help='Set the seed of the model')
parser.add_argument('--quant', type=int, default=0, help='no quant model') # quant=1:w8a8 
parser.add_argument('--strict', type=str, default=" ", help='force w4a4') # try: others w8a8 
parser.add_argument('--threshold', type=float, default=0.95, help='only calculate attn in mask')
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
strict_linear = strict

''' Add quantize '''
# def initialize_with_params_quant(cuda_devices, model_depth, mode, pic_num, seed, quant):
#     global CUDA_DEVICES, MODEL_DEPTH, MODE, PIC_NUM, SEED, QUANT
#     CUDA_DEVICES = cuda_devices
#     MODEL_DEPTH = model_depth
#     MODE = mode
#     PIC_NUM = pic_num
#     SEED = seed
#     QUANT = quant
#     # 根据参数设置环境变量
#     os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
# breakpoint()

if QUANT == 0:
    print(f'ERROR! no quantization!')
elif QUANT == 1:
    w_nbits = 8
    a_nbits = 8
    attn_nbits = 8
elif QUANT == 2:
    w_nbits = 4
    a_nbits = 4
    attn_nbits = 8
elif QUANT == 3:
    w_nbits = 6
    a_nbits = 6
    attn_nbits = 8
elif QUANT == 4:
    w_nbits = 4
    a_nbits = 8
    attn_nbits = 8
elif QUANT == 5:
    w_nbits = 4
    a_nbits = 6
    attn_nbits = 8
elif QUANT == 10:
    w_nbits = 8
    a_nbits = 8
    attn_nbits = 16
elif QUANT == 999:
    w_nbits = 4
    a_nbits = 4
    attn_nbits = 8

dim_1 = 1
dim_0 = 0
class QuantLinear(nn.Module):
    # breakpoint()
    def __init__(self, raw_layer, group_size, name=None):
        super().__init__()
        self.raw_layer = raw_layer
        self.group_size = group_size
        # weight quantizer
        # self.w_quantizer = BaseQuantizer(w_nbits)
        self.w_quantizer = WeightPCQuantizer(w_nbits)

        # self.w_quantizer = WeightPGQuantizer(self.group_size, w_nbits)
        # self.w_quantizer = WeightPCQuantileQuantizer(w_nbits)
        # self.w_quantizer = WeightPGQuantileQuantizer(w_nbits)

        # activation quantizer
        # self.a_quantizer = BaseQuantizer(a_nbits)
        # self.a_quantizer = ActDynamicQuantizer(a_nbits)
        self.a_quantizer = ActPGDynamicQuantizer(self.group_size, dim_1, a_nbits)
        
        self.name = name

    def forward(self, x):
        # breakpoint()
        if DEBUG:
            raw_out=F.linear(x,self.raw_layer.weight,self.raw_layer.bias)
        # breakpoint()
        w = self.w_quantizer(self.raw_layer.weight) # [11520, 1920]
        b = self.raw_layer.bias # 11520
        x_quant = self.a_quantizer(x) # [16, 1920]
        out = F.linear(x_quant, w, b)
        if DEBUG:
            layer_mse=F.mse_loss(raw_out,out)
            if torch.isnan(layer_mse).any():
                breakpoint()
            print(f"mse of layer {self.name} is {layer_mse}")
        return out

# 
class QuantLinear_w4a4(nn.Module):
    # breakpoint()
    def __init__(self, raw_layer, group_size, name=None):
        super().__init__()
        self.raw_layer = raw_layer
        self.group_size = group_size
        # self.w_quantizer = WeightPGQuantizer(self.group_size, 4)
        self.w_quantizer = WeightPCQuantizer(w_nbits)
        # 下面这行是QuantLinear_w4a4与QuantLinear的唯一区别：限定a_nbit=4
        self.a_quantizer = ActPGDynamicQuantizer(self.group_size, dim_1, 4)
        self.name = name

    def forward(self, x):
        if DEBUG:
            raw_out=F.linear(x,self.raw_layer.weight,self.raw_layer.bias)
        w = self.w_quantizer(self.raw_layer.weight)
        b = self.raw_layer.bias
        x_quant = self.a_quantizer(x)
        out = F.linear(x_quant, w, b)
        if DEBUG:
            layer_mse=F.mse_loss(raw_out,out)
            if torch.isnan(layer_mse).any():
                breakpoint()
            print(f"mse of layer {self.name} is {layer_mse}")
        return out
    
def identity(x):
    return x

#####################加入量化部分
class QuantSelfAttention(nn.Module):
    # breakpoint()
    def __init__(self, raw_layer, group_size):
        super().__init__()
        self.raw_layer = raw_layer
        self.group_size = group_size
        # self.qkv_weight_quantizer=WeightPGQuantizer(self.group_size, w_nbits)
        self.query_quantizer=ActPGDynamicQuantizer(self.group_size, dim_1, attn_nbits)
        self.key_quantizer=ActPGDynamicQuantizer(self.group_size, dim_1, attn_nbits)
        self.value_quantizer=ActDynamicQuantizer(attn_nbits,2)
        self.attn_map_quantizer=ActStaticScaleQuantizer(attn_nbits,-1)
        
        # self.qkv_weight_quantizer=identity
        # self.query_quantizer=identity
        # self.key_quantizer=lambda x:x
        # self.value_quantizer=lambda x:x
        # self.attn_map_quantizer=lambda x:x
    
    def forward(self, x, attn_bias):
        B, L, C = x.shape # 16 1 1920
        # breakpoint()
        
        raw_layer=self.raw_layer
        
        assert raw_layer.mat_qkv.raw_layer.bias is None
        pre_qkv=raw_layer.mat_qkv(x) # 16 1 5760
        bias=torch.cat((raw_layer.q_bias, raw_layer.zero_k_bias, raw_layer.v_bias)) # [5760]
        pre_qkv+=bias # 16 1 5760
        qkv=pre_qkv.view(B, L, 3, raw_layer.num_heads, raw_layer.head_dim) # [16, 1, 3, 30, 64]
        
        # quant_qkv_weight=self.qkv_weight_quantizer(raw_layer.mat_qkv.weight)
        
        # qkv = F.linear(input=x, weight=quant_qkv_weight, bias=torch.cat((raw_layer.q_bias, raw_layer.zero_k_bias, raw_layer.v_bias))).view(B, L, 3, raw_layer.num_heads, raw_layer.head_dim)
        main_type = qkv.dtype # fp16
        
        using_flash = raw_layer.using_flash and attn_bias is None and qkv.dtype != torch.float32
        if using_flash or raw_layer.using_xform: q, k, v = qkv.unbind(dim=2); dim_cat = 1   # q or k or v: BLHc
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); dim_cat = 2               # q or k or v: BHLc
        
        if raw_layer.attn_l2_norm:
            scale_mul = raw_layer.scale_mul_1H11.clamp_max(raw_layer.max_scale_mul).exp() # 1 30 1 1
            if using_flash or raw_layer.using_xform: scale_mul = scale_mul.transpose(1, 2)  # 1H11 to 11H1
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)

        if raw_layer.caching:
            if raw_layer.cached_k is None: raw_layer.cached_k = k; raw_layer.cached_v = v
            else: k = raw_layer.cached_k = torch.cat((raw_layer.cached_k, k), dim=dim_cat); v = raw_layer.cached_v = torch.cat((raw_layer.cached_v, v), dim=dim_cat)

        dropout_p = raw_layer.attn_drop if raw_layer.training else 0.0
        
        block_num = raw_layer.block_idx

        '''inference use below'''
        # breakpoint()
        if MODE == 0:
            # int8 q * int8 k
            quant_query=self.query_quantizer(q) # 16,30,1,64
            # breakpoint()
            quant_key=self.key_quantizer(k) # 16,30,1 64
            # breakpoint()
            if v.size(2)==1:
                quant_value=v
            else:
                quant_value=self.value_quantizer(v) # 16,30,1 64
            attn_mask=attn_bias
            quant_attn = quant_query.mul(raw_layer.scale) @ quant_key.transpose(-2, -1) # int16 # 16,30,1,1
            if attn_mask is not None: 
                quant_attn.add_(attn_mask)
            # 对int8 softmax & dropout
            # 先softmax
            softmax_attn_map = quant_attn.softmax(dim=-1)
            # 再量化
            if softmax_attn_map.size(-1)==1:
                quant_attention_map=softmax_attn_map
            else:
                quant_attention_map=self.attn_map_quantizer(softmax_attn_map) # int8 attentionmap 16 30 1 1
            # 再dropout（train）
            attention_map = (F.dropout(quant_attention_map, p=dropout_p, inplace=True) if dropout_p > 0 else quant_attention_map)
            
            # int8的attnmap * int8的value
            # attention_map:16 30 1 1  quant_value:16,30,1 64  
            quant_oup=(attention_map @ quant_value).transpose(1, 2).reshape(B, L, C) # 16 1 1920
            oup = quant_oup # int16
        

            # quant_attn = quant_cal_attnmap(query=quant_query, key=quant_key, scale=raw_layer.scale, attn_mask=attn_bias, dropout_p=dropout_p)
            # oup = slow_attn_original(query=quant_query, key=quant_key, value=v, block_num=block_num, scale=raw_layer.scale, attn_mask=attn_bias, dropout_p=dropout_p).transpose(1, 2).reshape(B, L, C)
        elif MODE == 1 or MODE == 2:
            # int8 q / int8 k/ int8 v/ int8 attn
            quant_query=self.query_quantizer(q)
            quant_key=self.key_quantizer(k)
            # 如果是B H 1 C 就不用量化了
            if v.size(2)==1:
                quant_value = v
            else:
                quant_value=self.value_quantizer(v)
            quant_attn=quant_query.mul(raw_layer.scale) @ quant_key.transpose(-2, -1) # int16
            # 先softmax
            softmax_attn_map = quant_attn.softmax(dim=-1)
            # 再量化
            if softmax_attn_map.size(-1)==1:
                quant_attention_map=softmax_attn_map
            else:
                quant_attention_map=self.attn_map_quantizer(softmax_attn_map) # int8 attentionmap 16 30 1 1
            # attn 是softmax后又量化的版本
            oup = quant_mask_inference(query=quant_query, key=quant_key, value=quant_value, attn=quant_attention_map,
                                    block_num=block_num, scale=raw_layer.scale, attn_mask=attn_bias, dropout_p=dropout_p
                                    ).transpose(1, 2).reshape(B, L, C)
        return raw_layer.proj(oup)
    
    def kv_caching(self,*args,**kwargs):
        output=self.raw_layer.kv_caching(*args,**kwargs)
        return output
    
    # def __setattr__(self, name: str, value: torch.Tensor | nn.Module) -> None:
    #     if name=='caching':
    #         self.raw_layer.caching=value
    #     print(f"======== DEBUG set {name} {value} ==========")
    #     return super().__setattr__(name, value)