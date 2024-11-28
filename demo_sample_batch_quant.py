################## 1. Download checkpoints and build models
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"

import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import matplotlib.pyplot as plt
import time
import tkinter as tk
# from models.quantize_all_me import WeightPCQuantizer, WeightPGQuantizer, WeightPCQuantileQuantizer, WeightPGQuantileQuantizer
# from models.quantize_all_me import BaseQuantizer, ActDynamicQuantizer, ActPGDynamicQuantizer
import argparse
import torch.nn as nn
import torch.nn.functional as F
# breakpoint()
# from models.mask_for_inference import mask_inference, quant_mask_inference
# from models.basic_var import slow_attn_original
# from models.basic_var import uantSelfAttention
DEBUG=False
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--cuda_devices', type=str, default="4,", help='Specify the CUDA devices to use, e.g., "0,1" for using devices 0 and 1')
parser.add_argument('--model_depth', type=int, choices=[16, 20, 24, 30], required=True, help='Specify the model depth')
parser.add_argument('--mode', type=int, choices=[0, 1, 2, -1], required=True, help='Specify the mode: 0 for original, 1 for masked, 2 for global cfg, -1 for design mask')
parser.add_argument('--pic_num', type=int, default=250, help='Specify the number of images to generate')
parser.add_argument('--seed', type=int, default=0, help='Set the seed of the model')
parser.add_argument('--quant', type=int, default=0, help='no quant model') # quant=1:w8a8 
parser.add_argument('--strict', type=str, default=" ", help='force w4a4') # try: others w8a8 
parser.add_argument('--threshold', type=float, default=0.95, help='only calculate attn in mask')

args = parser.parse_args()

cuda_devices = args.cuda_devices
MODEL_DEPTH = args.model_depth
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
MODE = args.mode
PIC_NUM = args.pic_num
seed = args.seed
QUANT = args.quant
strict = args.strict
threshold = args.threshold
print(f"Running with the following parameters:")
print(f"Model depth: {args.model_depth}")
print(f"Mode: {args.mode}")
print(f"Number of images: {args.pic_num}")
print(f"Seed: {args.seed}")
print(f"Quant: {args.quant}")
print(f"Threshold: {args.threshold}")

params = {
    'cuda_devices': cuda_devices,
    'model_depth': args.model_depth,
    'mode': args.mode,
    'pic_num': args.pic_num,
    'seed': args.seed,
    'quant': args.quant,
    'strict': args.strict,
    'threshold': args.threshold
}
# strict_linear_list = ["word_embed", "attn.mat_qkv", "attn.proj", "ffn.fc1", "ffn.fc2", "ada_lin.1", "head"]

# strict_linear = strict_linear_list[strict]
strict_name = strict
# breakpoint()
# for block_idx in range(0,30):

#     strict_linear = f'blocks.{block_idx}.{strict_name}'
#     print(f'###################### Now strict linear is {strict_linear} ##########################')
strict_linear = strict_name
from models import quant_self_attention
# quant_self_attention.initialize_with_params_quant(**params)

from models import VQVAE, build_vae_var, basic_var
# basic_var.initialize_with_params(**params)
from models.basic_var import SelfAttention



from models import mask_for_inference
# mask_for_inference.initialize_with_params_mask(**params)
    

print(f"CUDA device#####################################################: {args.cuda_devices}")

data_path = "/share/xierui-nfs/dataset/imageNet-1k/"
labels_path = os.path.join(data_path, 'labels.txt')
traindir = os.path.join(data_path, 'train')
valdir = os.path.join(data_path, 'val')


# download checkpoint
hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{MODEL_DEPTH}.pth'
home_path = '/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/'

my_vae_ckpt = os.path.join(home_path, 'vae_ch160v4096z32.pth')
my_var_ckpt =  os.path.join(home_path, f'var_d{MODEL_DEPTH}.pth')
if not osp.exists(my_vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
if not osp.exists(my_var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')

# build vae, var
patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using GPU: {torch.cuda.current_device()}') if device == 'cuda' else print('Using CPU')
# if 'vae' not in globals() or 'var' not in globals():
#     vae, var = build_vae_var(
#         V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
#         device=device, patch_nums=patch_nums,
#         num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
#     )
vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
        device=device, patch_nums=patch_nums,
        num_classes=1000, depth=MODEL_DEPTH, shared_aln=False,
    )
# import ipdb
# ipdb.set_trace()

# load checkpoints
vae.load_state_dict(torch.load(my_vae_ckpt, map_location='cpu'), strict=True)
vae.eval()
# print('VAE', torch.cuda.max_memory_allocated()/(1024*1024*1024), 'GB')
var.load_state_dict(torch.load(my_var_ckpt, map_location='cpu'), strict=True)
var.eval()
var_copy = var
var_copy.eval()
# print('VAR', torch.cuda.max_memory_allocated()/(1024*1024*1024), 'GB')
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
for p in var_copy.parameters(): p.requires_grad_(False)
print(f'prepare finished.')


############## quantization repalcement


############################# 2. Sample with classifier-free guidance

# set args
# seed = 1 #@param {type:"number"}
from models.quant_self_attention import QuantSelfAttention, QuantLinear, QuantLinear_w4a4


strict_linear_list = ["word_embed", "attn.mat_qkv", "attn.proj", "ffn.fc1", "ffn.fc2", "ada_lin.1", "head"]
''''''
# for strict_linear in strict_linear_list:

# check if need replacement
def should_replace(name):
    # dont want to replace
    skip_list = ['blocks.1.ffn.fc1']
    return name not in skip_list

def replace_to_quantize_layer_strict(our_net, strict_layer, parent_name=''):
    for name, child in our_net.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        # print(f"now full name----{full_name}")
        if isinstance(child, nn.Linear):
            # skip one kind of layer
            if strict_layer in full_name:
                hidden_size = child.in_features
                quant_layer = QuantLinear_w4a4(child, hidden_size)
                setattr(our_net, name, quant_layer)
                print(f'replace {full_name} to {quant_layer}')
            
            else:
                hidden_size = child.in_features
                quant_layer = QuantLinear(child, hidden_size)
                setattr(our_net, name, quant_layer)
                print(f'replace {full_name} to {quant_layer}')

        elif isinstance(child, SelfAttention):
            head_dim = child.head_dim
            quant_layer = QuantSelfAttention(child, head_dim)
            setattr(our_net, name, quant_layer)
            print(f'replace {full_name} to {quant_layer}')
        replace_to_quantize_layer_strict(child, strict_layer, full_name)

def replace_to_quantize_layer(our_net, parent_name=''):
    for name, child in our_net.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear):
            # if "mat_qkv" in name:
            #     continue
            # if 'word_embed' in name:
            #     continue
            if "ffn.fc2" in full_name:
                continue
            hidden_size = child.in_features
            quant_layer = QuantLinear(child, hidden_size)
            setattr(our_net, name, quant_layer)
            print(f'replace {full_name} to {quant_layer}')
        elif isinstance(child,SelfAttention):
            head_dim = child.head_dim
            quant_layer = QuantSelfAttention(child, head_dim)
            setattr(our_net, name, quant_layer)
            print(f'replace {full_name} to {quant_layer}')
        replace_to_quantize_layer(child, full_name)

def replace_to_quantize_layer_only_linear(our_net, parent_name=''):
    for name, child in our_net.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(child, nn.Linear):
            hidden_size = child.in_features
            quant_layer = QuantLinear(child, hidden_size)
            setattr(our_net, name, quant_layer)
            print(f'replace {full_name} to {quant_layer}')
        replace_to_quantize_layer_only_linear(child, full_name)  

'''no quant'''
if QUANT == 0:
    torch.manual_seed(seed)
    cfg = 4

    class_labels_map = {}
    start_time = time.time()


    for pic in range(0, PIC_NUM):
        if MODE == -1:
            print(f"####### mode = -1 #######")
            random_numbers = random.sample(range(1000), 8)
            class_labels = tuple(random_numbers)
            print(f"pic {pic} class label is: {class_labels}")
        else:
            print(f"NOW IMAGE —— {pic}")
            # class_labels_map[pic] = tuple([item for j in range(4) for item in (j + 4 * pic, j + 4 * pic)])
            # {0: (0, 0, 1, 1, 2, 2, 3, 3), 1: (4, 4, 5, 5, 6, 6, 7, 7),...249: (996, 996, 997, 997, 998, 998, 999, 999)}
            class_labels_map[pic] = tuple([item for j in range(4) for item in (pic, pic)]) # 00000000 11111111 22222222 ……
            class_labels = class_labels_map[pic]

        more_smooth = False # True for more smooth output

        # seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # run faster
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        # sample
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=B, 
                                                        label_B=label_B, 
                                                        cfg=cfg, 
                                                        top_k=900, 
                                                        top_p=0.95, 
                                                        g_seed=seed, 
                                                        more_smooth=more_smooth,
                                                        #   model_depth=args.model_depth,
                                                        #   mode=args.mode
                                                        )

        chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0) # 每行8张图，图像无间隔，pad_value=1.0表示使用白色填充图像间的空白区域
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy() # 因为PIL.Image.fromarray函数默认使用通道顺序为RGB。将像素值从[0, 1]的范围映射到[0, 255]。
        chw = PImage.fromarray(chw.astype(np.uint8)) # 用PIL.Image.fromarray函数将NumPy数组chw转换为PIL.Image.Image对象。它将NumPy数组作为参数，并指定数据类型为np.uint8
        # breakpoint()
        # if MODE == 0: # MODE = 0 
        #     original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/fp16/all_original/'
        #     os.makedirs(original_savedir, exist_ok=True)
        #     chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{pic}.png')
        # elif MODE == 1: # MODE = 1 
        #     masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/fp16/all_masked/'
        #     os.makedirs(masked_savedir, exist_ok=True)
        #     chw.save(masked_savedir+f'demo_{MODEL_DEPTH}_masked_{pic}.png')
        # elif MODE == 2: # cfg
        #     cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/fp16/all_cfg/'
        #     os.makedirs(cfg_savedir, exist_ok=True)
        #     chw.save(cfg_savedir+f'demo_{MODEL_DEPTH}_masked_cfg_{pic}.png')
        if MODE == 0: # MODE = 0
            original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/fp16/all_original/'
            os.makedirs(original_savedir, exist_ok=True)
            chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{pic}.png')
        ##################### slice picture (optional)
        # width, height = chw.size

        # new_height = height
        # column_width = width // 8

        # # slice and save each column
        # img_list_1 = []
        # for single_pic in range(B):
        #     left = single_pic * column_width
        #     upper = 0
        #     right = left + column_width
        #     lower = new_height

        #     column_img = chw.crop((left, upper, right, lower))
        #     if MODE == 0: # MODE = 0 
        #         original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/fp16/all_original/'
        #         os.makedirs(original_savedir, exist_ok=True)
        #         single_name = f"single_{MODEL_DEPTH}_original_{pic*8+single_pic}.png"   
        #         check_name = f"single_{MODEL_DEPTH}_original_7999.png"
        #         column_img.save(original_savedir+single_name) 
                
        #     elif MODE == 1: # MODE = 1 
        #         masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/fp16/all_masked/'
        #         os.makedirs(masked_savedir, exist_ok=True)
        #         single_name = f"single_{MODEL_DEPTH}_masked_{pic*8+single_pic}.png"   
        #         check_name = f"single_{MODEL_DEPTH}_masked_7999.png"
        #         column_img.save(masked_savedir+single_name) 
        #     elif MODE == 2: #
        #         cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/fp16/all_cfg/'
        #         os.makedirs(cfg_savedir, exist_ok=True)
        #         single_name = f"single_{MODEL_DEPTH}_masked_cfg_{pic*8+single_pic}.png"   
        #         check_name = f"single_{MODEL_DEPTH}_masked_cfg_7999.png"
        #         column_img.save(cfg_savedir+single_name) 

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print(f"MODE = {MODE}")
    print(f"QUANT = {QUANT}")


else:
    '''quant'''
    if QUANT == 1:
        save_name = 'w8a8attn8'
    elif QUANT == 2:
        save_name = 'w4a4attn8'
    elif QUANT == 3:
        save_name = 'w6a6attn8'
    elif QUANT == 4:
        save_name = 'w4a8attn8'
    elif QUANT == 5:
        save_name = 'w4a6attn8'
    elif QUANT == 10:
        save_name = 'w8a8attn16'
    elif QUANT == 999:
        save_name = 'try_sth'

    # no quantization in QKV:
    if QUANT == 10:
        replace_to_quantize_layer_only_linear(var)
    else:
        replace_to_quantize_layer(var)
        # print(var)
        # strict some layer
        # replace_to_quantize_layer_strict(var, strict_layer=strict_linear)

        # else:
        #     replace_to_quantize_layer_strict(var, strict_layer=strict_linear)
    # breakpoint()

    for name, module in var.named_modules():
        if isinstance(module, QuantLinear):
            module.w_quantizer.do_calibration = True
            module.a_quantizer.do_calibration = True
            module.w_quantizer.name = f'calib {name} wq'
            module.a_quantizer.name = f'calib {name} aq'
        if isinstance(module, QuantLinear_w4a4):
            module.w_quantizer.do_calibration = True
            module.a_quantizer.do_calibration = True
            module.w_quantizer.name = f'calib {name} wq'
            module.a_quantizer.name = f'calib {name} aq'
        if isinstance(module, QuantSelfAttention):
            # module.qkv_weight_quantizer.do_calibration=True
            module.query_quantizer.do_calibration=True
            module.key_quantizer.do_calibration=True
            module.value_quantizer.do_calibration=True
            module.attn_map_quantizer.do_calibration=True
        module.name=name

    torch.manual_seed(seed)
    cfg = 4 
    class_labels_map = {}
    start_time = time.time()

    # 生成 100 个映射
    '''calibration'''
    calib_indices = [10, 50, 88, 100, 150, 187, 272, 380, 429, 480, 522, 597, 630, 680, 750, 790, 850, 880, 900, 972]     
    for calib_idx in calib_indices:
        print(f"NOW IMAGE —— {calib_idx}")
        class_labels_map[calib_idx] = tuple((calib_idx + j for j in range(8)))
        class_labels = class_labels_map[calib_idx]

        more_smooth = False # True for more smooth output

        # seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # run faster
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        # sample
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=B, 
                                                        label_B=label_B, 
                                                        cfg=cfg, 
                                                        top_k=900, 
                                                        top_p=0.95, 
                                                        g_seed=seed, 
                                                        more_smooth=more_smooth,
                                                        #   model_depth=args.model_depth,
                                                        #   mode=args.mode
                                                        )
                
    for name, module in var.named_modules():
        if isinstance(module, QuantLinear):
            module.w_quantizer.do_calibration = False
            module.a_quantizer.do_calibration = False
        if isinstance(module, QuantLinear_w4a4):
            module.w_quantizer.do_calibration = False
            module.a_quantizer.do_calibration = False
        if isinstance(module, QuantSelfAttention):
            # module.qkv_weight_quantizer.do_calibration=False
            module.query_quantizer.do_calibration=False
            module.key_quantizer.do_calibration=False
            module.value_quantizer.do_calibration=False
            module.attn_map_quantizer.do_calibration=False

    '''calibration end'''

    for pic in range(0, PIC_NUM):
        # start_time = time.time()
        print(f"NOW IMAGE —— {pic}")
        # class_labels_map[pic] = tuple([item for j in range(4) for item in (j + 4 * pic, j + 4 * pic)])
        # {0: (0, 0, 1, 1, 2, 2, 3, 3), 1: (4, 4, 5, 5, 6, 6, 7, 7),...249: (996, 996, 997, 997, 998, 998, 999, 999)}
        class_labels_map[pic] = tuple([item for j in range(4) for item in (pic, pic)])
        class_labels = class_labels_map[pic]

        more_smooth = False # True for more smooth output

        # seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # run faster
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')

        # sample
        B = len(class_labels)
        label_B: torch.LongTensor = torch.tensor(class_labels, device=device)
        
        with torch.inference_mode():
            with torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):    # using bfloat16 can be faster
                recon_B3HW = var.autoregressive_infer_cfg(B=B, 
                                                        label_B=label_B, 
                                                        cfg=cfg, 
                                                        top_k=900, 
                                                        top_p=0.95, 
                                                        g_seed=seed, 
                                                        more_smooth=more_smooth,
                                                        #   model_depth=args.model_depth,
                                                        #   mode=args.mode
                                                        )

        chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0) 
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy() 
        chw = PImage.fromarray(chw.astype(np.uint8)) 
        '''naive quant'''
        # if MODE == 0: # MODE = 0 original image
        #     original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{save_name}/all_original/'
        #     os.makedirs(original_savedir, exist_ok=True)
        #     chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{pic}.png')
        # elif MODE == 1: # MODE = 1 mask
        #     masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/quant/{save_name}/all_masked/'
        #     os.makedirs(masked_savedir, exist_ok=True)
        #     chw.save(masked_savedir+f'demo_{MODEL_DEPTH}_masked_{pic}.png')
        # elif MODE == 2: # cfg
        #     cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/quant/{save_name}/all_cfg/'
        #     os.makedirs(cfg_savedir, exist_ok=True)
        #     chw.save(cfg_savedir+f'demo_{MODEL_DEPTH}_masked_cfg_{pic}.png')
        '''mix percision'''
        # if MODE == 0: # MODE = 0 
        #     original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/'
        #     os.makedirs(original_savedir, exist_ok=True)
        #     chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{pic}.png')
        # elif MODE == 1: # MODE = 1 
        #     masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/'
        #     os.makedirs(masked_savedir, exist_ok=True)
        #     chw.save(masked_savedir+f'demo_{MODEL_DEPTH}_masked_{pic}.png')
        # elif MODE == 2: 
        #     cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/'
        #     os.makedirs(cfg_savedir, exist_ok=True)
        #     chw.save(cfg_savedir+f'demo_{MODEL_DEPTH}_masked_cfg_{pic}.png')
        '''draw picture'''
        if MODE == 0: # MODE = 0
            original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/'
            os.makedirs(original_savedir, exist_ok=True)
            chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{pic}.png')
        elif MODE == 1: # MODE = 1 
            masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/'
            os.makedirs(masked_savedir, exist_ok=True)
            chw.save(masked_savedir+f'demo_{MODEL_DEPTH}_masked_{pic}.png')
        elif MODE == 2: # 
            cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/'
            os.makedirs(cfg_savedir, exist_ok=True)
            chw.save(cfg_savedir+f'demo_{MODEL_DEPTH}_masked_cfg_{pic}.png')


        ##################### slice and save 
        # width, height = chw.size

        # new_height = height
        # column_width = width // 8

        # img_list_1 = []
        # for single_pic in range(B):
        #     # 计算每一列的左上角和右下角坐标
        #     left = single_pic * column_width
        #     upper = 0
        #     right = left + column_width
        #     lower = new_height

        #     # 切割图像并保存
        #     column_img = chw.crop((left, upper, right, lower))
        #     if MODE == 0: 
        #         original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{save_name}/all_original/'
        #         os.makedirs(original_savedir, exist_ok=True)
        #         single_name = f"single_{MODEL_DEPTH}_original_{pic*8+single_pic}.png"   
        #         check_name = f"single_{MODEL_DEPTH}_original_7999.png"
        #         column_img.save(original_savedir+single_name)
                
        #     elif MODE == 1: 
        #         masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{save_name}/all_masked/'
        #         os.makedirs(masked_savedir, exist_ok=True)
        #         single_name = f"single_{MODEL_DEPTH}_masked_{pic*8+single_pic}.png"   
        #         check_name = f"single_{MODEL_DEPTH}_masked_7999.png"
        #         column_img.save(masked_savedir+single_name) 
        #     elif MODE == 2: 
        #         cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{save_name}/all_cfg/'
        #         os.makedirs(cfg_savedir, exist_ok=True)
        #         single_name = f"single_{MODEL_DEPTH}_masked_cfg_{pic*8+single_pic}.png"   
        #         check_name = f"single_{MODEL_DEPTH}_masked_cfg_7999.png"
        #         column_img.save(cfg_savedir+single_name)


        '''ban one kind of linear layer'''
        # if MODE == 0:
        #     original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{strict_name}/{save_name}/all_original/'
        #     os.makedirs(original_savedir, exist_ok=True)
        #     chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{pic}.png')
        # elif MODE == 1:
        #     masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{strict_name}/{save_name}/all_masked/'
        #     os.makedirs(masked_savedir, exist_ok=True)
        #     chw.save(masked_savedir+f'demo_{MODEL_DEPTH}_masked_{pic}.png')
        # elif MODE == 2: 
        #     # cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/test0/all_cfg/'
        #     cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/quant/{strict_name}/{save_name}/all_cfg/'
        #     os.makedirs(cfg_savedir, exist_ok=True)
        #     chw.save(cfg_savedir+f'demo_{MODEL_DEPTH}_masked_cfg_{pic}.png')

        '''ban one layer in every block'''
        # if MODE == 0:
        #     original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/quant/detail/{strict_name}/block_{block_idx}/{save_name}/all_original/'
        #     os.makedirs(original_savedir, exist_ok=True)
        #     chw.save(original_savedir+f'd_{MODEL_DEPTH}_original_{pic}.png')
        # elif MODE == 1:
        #     masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/quant/detail/{strict_name}/block_{block_idx}/{save_name}/all_masked/'
        #     os.makedirs(masked_savedir, exist_ok=True)
        #     chw.save(masked_savedir+f'd_{MODEL_DEPTH}_masked_{pic}.png')
        # elif MODE == 2:
        #     # cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/test0/all_cfg/'
        #     cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/quant/detail/{strict_name}/block_{block_idx}/{save_name}/all_cfg/'
        #     os.makedirs(cfg_savedir, exist_ok=True)
        #     chw.save(cfg_savedir+f'd_{MODEL_DEPTH}_masked_cfg_{pic}.png')

    #     end_time = time.time()
    #     execution_time = end_time - start_time
    #     print("Execution time:", execution_time, "seconds")
    # print(f"MODE = {MODE}")

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print(f"MODE = {MODE}")
    print(f"QUANT = {QUANT}")
    # print(f"strict_layer = {strict_linear}")
