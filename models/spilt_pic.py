# 把一行8张的图拆成8张

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
from PIL import Image
import ImageReward as RM
import time
import json
import argparse

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

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''quant'''
# if QUANT == 0:
#     save_name = 'fp16'
# if QUANT == 1:
#     save_name = 'quant/w8a8attn8'
# elif QUANT == 2:
#     save_name = 'quant/w4a4attn8'
# elif QUANT == 3:
#     save_name = 'quant/w6a6attn8'
# elif QUANT == 4:
#     save_name = 'quant/w4a8attn8'
# elif QUANT == 5:
#     save_name = 'quant/w4a6attn8'
# elif QUANT == 10:
#     save_name = 'quant/w8a8attn16'
# elif QUANT == 999:
#     save_name = 'quant/try_sth'
if QUANT == 0:
    save_name = 'fp16'
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
# if QUANT == 1:
#     save_name = f'quant/{strict_linear}/w8a8attn8'

# 原图：/share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/fp16/all_original/demo_30_original_0.png 已测
# 原图+0.95mask /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/threshold_0.95/fp16/all_masked 已测
# 原图+0.95mask+cfg /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/threshold_0.95/fp16/all_cfg 已测

# 量化w8a8+原图 /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/quant/w8a8attn8/all_original/demo_30_original_0.png
# 量化w8a8+原图+mask /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/threshold_0.95/quant/w8a8attn8/all_masked/demo_30_masked_0.png
# 量化w8a8+原图+mask+cfg /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/threshold_0.95/quant/w8a8attn8/all_cfg/demo_30_masked_cfg_0.png

for i in [14, 22]:
    # print(f"NOW IMAGE —— {i}")
    # if MODE == 0: # 原图
    #     # img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
    # elif MODE == 1: # 仅严格mask
    #     # img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")

    # elif MODE == 2: # 全局reuse
    #     # img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
    
    # mix-percision
    # if MODE == 0: # MODE = 0 原图，无任何操作
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
    # elif MODE == 1: # MODE = 1 不合并cond和uncond，仅全scale严格mask
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
    # elif MODE == 2: # 目前采取全局cfg合并
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
    # strict layer
    # if MODE == 0: # MODE = 0 原图，无任何操作
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
    # 好的strict layer
    # if MODE == 0: # MODE = 0 原图，无任何操作
    #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_original/d_{MODEL_DEPTH}_original_{i}.png")
    if MODE == 0: # MODE = 0 原图，无任何操作/share/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_0/mix-percision/w4a8attn8/fc2-16/all_original/demo_30_original_14.png
        img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
    elif MODE == 1: # MODE = 1 不合并cond和uncond，仅全scale严格mask /share/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_0/threshold_0.95/mix-percision/w4a4attn8/fc2-16/all_masked/demo_30_masked_0.png
        img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
    elif MODE == 2: # 目前采取全局cfg合并
        img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
    width, height = img.size

    new_height = height
    column_width = width // 8

    # 切割并保存每一列的图像
    img_list_1 = []
    for pic in range(8):
        # 计算每一列的左上角和右下角坐标
        left = pic * column_width
        upper = 0
        right = left + column_width
        lower = new_height
        
        # 切割图像并保存
        # column_img = img.crop((left, upper, right, lower))
        # if MODE == 0:
        #     # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/single/seed_{seed}/{save_name}/all_original/"
        #     single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/{save_name}/all_original/"
            
        #     os.makedirs(single_dir, exist_ok=True)
        #     single_name = f"single_{MODEL_DEPTH}_original_{i*8+pic}.png"   
        #     check_name = f"single_{MODEL_DEPTH}_original_7999.png"
        # elif MODE == 1:
        #     # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/single/seed_{seed}/{save_name}/all_masked/"
        #     single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/{save_name}/all_masked/"
            
        #     os.makedirs(single_dir, exist_ok=True)
        #     single_name = f"single_{MODEL_DEPTH}_masked_{i*8+pic}.png"
        #     check_name = f"single_{MODEL_DEPTH}_masked_7999.png"
        # elif MODE == 2:
        #     # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/single/seed_{seed}/{save_name}/all_cfg/"
        #     single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/{save_name}/all_cfg/"
            
        #     os.makedirs(single_dir, exist_ok=True)
        #     single_name = f"single_{MODEL_DEPTH}_cfg_{i*8+pic}.png"
        #     check_name = f"single_{MODEL_DEPTH}_cfg_7999.png"
        # column_img.save(single_dir+single_name)  # 保存每一列为单独的图像


        # mix percision
        # column_img = img.crop((left, upper, right, lower))
        # if MODE == 0:
        #     # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/single/seed_{seed}/{save_name}/all_original/"
        #     single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/" 
        #     os.makedirs(single_dir, exist_ok=True)
        #     single_name = f"single_{MODEL_DEPTH}_original_{i*8+pic}.png"   
        #     check_name = f"single_{MODEL_DEPTH}_original_7999.png"
        # elif MODE == 1:
        #     # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/single/seed_{seed}/{save_name}/all_masked/"
        #     single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/"
        #     os.makedirs(single_dir, exist_ok=True)
        #     single_name = f"single_{MODEL_DEPTH}_masked_{i*8+pic}.png"
        #     check_name = f"single_{MODEL_DEPTH}_masked_7999.png"
        # elif MODE == 2:
        #     single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/"
        #     os.makedirs(single_dir, exist_ok=True)
        #     single_name = f"single_{MODEL_DEPTH}_cfg_{i*8+pic}.png"
        #     check_name = f"single_{MODEL_DEPTH}_cfg_7999.png"
        # column_img.save(single_dir+single_name)  # 保存每一列为单独的图像
        # strict layer 
        column_img = img.crop((left, upper, right, lower))
        if MODE == 0:
            # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/{save_name}/all_original/" 
            single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/single/seed_{seed}/{save_name}/all_original/" 
            
            os.makedirs(single_dir, exist_ok=True)
            single_name = f"single_{MODEL_DEPTH}_original_{i*8+pic}.png"   
        elif MODE == 1:
            # single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/single/seed_{seed}/{save_name}/all_masked/"
            single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/single/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/"
            os.makedirs(single_dir, exist_ok=True)
            single_name = f"single_{MODEL_DEPTH}_masked_{i*8+pic}.png"
            # check_name = f"single_{MODEL_DEPTH}_masked_7999.png"
        elif MODE == 2:
            single_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/draw_pic/output_pic/single/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/"
            os.makedirs(single_dir, exist_ok=True)
            single_name = f"single_{MODEL_DEPTH}_cfg_{i*8+pic}.png"
            # check_name = f"single_{MODEL_DEPTH}_cfg_7999.png"


            # check_name = f"single_{MODEL_DEPTH}_original_1999.png"
        column_img.save(single_dir+single_name)  # 保存每一列为单独的图像

# file_check = single_dir+check_name
# if os.path.exists(file_check):
#     print("All is well")
# else:
#     raise FileNotFoundError("ERROR: pic_7999 not exist!")
