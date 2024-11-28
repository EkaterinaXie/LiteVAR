################## 1. Download checkpoints and build models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import matplotlib.pyplot as plt
import time
import tkinter as tk

setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
from global_vars import MODEL_DEPTH, MODE, PIC_NUM, seed

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
if 'vae' not in globals() or 'var' not in globals():
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
# print('VAR', torch.cuda.max_memory_allocated()/(1024*1024*1024), 'GB')
for p in vae.parameters(): p.requires_grad_(False)
for p in var.parameters(): p.requires_grad_(False)
print(f'prepare finished.')

############################# 2. Sample with classifier-free guidance

# set args
# seed = 1 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 250 #@param {type:"slider", min:0, max:1000, step:1}
cfg = 4 #@param {type:"slider", min:1, max:10, step:0.1}

# 初始化一个空的 class_labels_map 字典
class_labels_map = {}


# 生成 100 个映射
for i in range(PIC_NUM):
    # 计算耗时
    start_time = time.time()
    print(f"NOW IMAGE —— {i}")
    
    class_labels_map[i] = tuple([item for j in range(4) for item in (j + 4 * i, j + 4 * i)])
    # {0: (0, 0, 1, 1, 2, 2, 3, 3), 1: (4, 4, 5, 5, 6, 6, 7, 7),...249: (996, 996, 997, 997, 998, 998, 999, 999)}
    class_labels = class_labels_map[i]

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
            recon_B3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=seed, more_smooth=more_smooth)

    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0) # 每行8张图，图像无间隔，pad_value=1.0表示使用白色填充图像间的空白区域
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy() # 因为PIL.Image.fromarray函数默认使用通道顺序为RGB。将像素值从[0, 1]的范围映射到[0, 255]。
    # 经过这些操作，变量chw将成为一个表示图像的NumPy数组，像素值范围为[0, 255]。
    chw = PImage.fromarray(chw.astype(np.uint8)) # 用PIL.Image.fromarray函数将NumPy数组chw转换为PIL.Image.Image对象。它将NumPy数组作为参数，并指定数据类型为np.uint8
    # chw.show() # 用show()方法来显示PIL.Image.Image对象chw中的图像。show()方法会打开一个图像查看器，显示图像内容。

    if MODE == 0: # MODE = 0 原图，无任何操作
        original_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/all_original/'
        os.makedirs(original_savedir, exist_ok=True)
        chw.save(original_savedir+f'demo_{MODEL_DEPTH}_original_{i}.png')
    elif MODE == 1: # MODE = 1 不合并cond和uncond，仅全scale严格mask
        masked_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/all_masked/'
        os.makedirs(masked_savedir, exist_ok=True)
        chw.save(masked_savedir+f'demo_{MODEL_DEPTH}_masked_{i}.png')
    elif MODE == 2: # 目前采取全局cfg合并
        cfg_savedir = f'/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/all_cfg/'
        os.makedirs(cfg_savedir, exist_ok=True)
        chw.save(cfg_savedir+f'demo_{MODEL_DEPTH}_masked_cfg_{i}.png')

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
print(f"MODE = {MODE}")