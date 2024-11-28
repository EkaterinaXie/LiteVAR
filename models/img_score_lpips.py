# lpips指标
# 多张图
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
import lpips
import torchvision
import numpy as np
from PIL import Image
import time
import argparse
import pdb
# MODEL_DEPTH = 30
# PIC_NUM = 2
# MODE = 1 # 0:原图 1:仅严格mask 2:cfg的全局reuse
# seed = 0
# assert MODE in {1, 2}
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
print(f"Running with the following parameters:")
print(f'Cuda: {cuda_devices}')
print(f"Model depth: {args.model_depth}")
print(f"Mode: {args.mode}")
print(f"Number of images: {args.pic_num}")
print(f"Seed: {args.seed}")
print(f"Quant: {args.quant}")
print(f"Strict layer: {args.strict}")

# 检查 CUDA 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# 计算耗时
start_time = time.time()

# 初始化 LPIPS 模型
'''
LPIPS 模型 只能对比mask/cfg后VS原图
'''
# model_weights_path = '/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/pth_download/checkpoints/alexnet-owt-7be5be79.pth'

# # 检查模型权重文件是否已经存在
# if not os.path.exists(model_weights_path):
#     # 设置自定义的模型下载路径
#     torch.hub.set_dir('/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/pth_download')
#     # 使用 LPIPS
#     loss_fn = lpips.LPIPS(net='alex').to(device) 
# else:
#     # 使用已存在的模型权重文件
#     loss_fn = lpips.LPIPS(net='alex').to(device) 

# loss_fn = lpips.LPIPS(net='alex')
# 设置自定义的模型下载路径
torch.hub.set_dir('/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/pth_download')
# 使用 LPIPS
loss_fn = lpips.LPIPS(net='alex').cuda()


'''quant'''
if QUANT== 0:
    save_name = 'fp16'
if QUANT == 1:
    save_name = 'quant/w8a8attn8'
elif QUANT == 2:
    save_name = 'quant/w4a4attn8'
elif QUANT == 3:
    save_name = 'quant/w6a6attn8'
elif QUANT == 4:
    save_name = 'quant/w4a8attn8'
elif QUANT == 5:
    save_name = 'quant/w4a6attn8'
elif QUANT == 10:
    save_name = 'quant/w8a8attn16'
elif QUANT == 999:
    save_name = 'quant/try_sth'
if QUANT == 1:
    save_strict_name = f'quant/{strict_linear}/w8a8attn8'

# Load the saved images
score = {}
for i in range(PIC_NUM):
    print(f"IMAGE —— {i}")
    # breakpoint()
    if strict_linear != " ":
        if MODE == 0: # 仅严格mask
            img1_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_strict_name}/all_original/d_{MODEL_DEPTH}_original_{i}.png"
            img1 = Image.open(img1_path)
            img2_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/fp16/all_original/demo_{MODEL_DEPTH}_original_{i}.png"
            img2 = Image.open(img2_path)
    else:
        assert MODE == 1 or MODE == 2, "MODE should be 1 or 2"
        if MODE == 1: # 仅严格mask
            img1_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png"
            img1 = Image.open(img1_path)
            img2_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/fp16/all_original/demo_{MODEL_DEPTH}_original_{i}.png"
            img2 = Image.open(img2_path)

        if MODE == 2: # 全局reuse
            img1_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png"
            img1 = Image.open(img1_path)
            img2_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/fp16/all_original/demo_{MODEL_DEPTH}_original_{i}.png"
            img2 = Image.open(img2_path)

    # Convert images to PyTorch tensors
    img1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # img3_tensor = torch.tensor(np.array(img3)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # print(f'#########################{img1_tensor.device}')
    # Reshape the images to get individual images
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    # img3_tensor = img3_tensor.to(device)
    # print(f'#########################{img1_tensor.device}')
    img1_split = torch.split(img1_tensor, img1_tensor.shape[3] // 8, dim=3)
    img2_split = torch.split(img2_tensor, img2_tensor.shape[3] // 8, dim=3)
    # img3_split = torch.split(img3_tensor, img3_tensor.shape[3] // 8, dim=3)

    lpips_scores = []
        
    # Calculate LPIPS score between corresponding images in img1_split and img2_split
    for img1, img2 in zip(img1_split, img2_split):
        # breakpoint()
        # print(f'#########################{img1.device}')
        # print(loss_fn(img1, img2).item().device)
        # print(loss_fn(img1, img2).item().device)
        lpips_score = loss_fn(img1, img2).item()
        
        # print(f"{lpips_score:.4f}")
        lpips_scores.append(lpips_score)
    score[i] = lpips_scores
    print(lpips_scores)
# print(f'SCORE OF EACH PICTURE: {score}')
# print(score)

score_one_group = []
score_one_group_mean = []
# 一组图像包括八张
for index in range(8):
    # 初始化总和
    total_sum = 0
    # 将每个键对应列表的对应元素相加
    for key in range(PIC_NUM):
        total_sum += score[key][index]
    score_one_group.append(total_sum)
print(f'sum lpips of all pics: {score_one_group}')
# 将列表中的每个值都除以给定的数
score_one_group_mean = [num / PIC_NUM for num in score_one_group]

# print(f'mean lpips of all pictures: {score_one_group_mean}')

for index in range(len(score_one_group_mean)):
    mean_each_place = score_one_group_mean[index]
    mean_of_pic_0 = score_one_group_mean[index]
    print(f"mean LPIPS of the {index} place: {mean_each_place:.4f}")

# 奇数索引元素相加
mean_odd_index = sum(score_one_group_mean[1::2]) / 4
# 偶数索引元素相加
mean_even_index = sum(score_one_group_mean[::2]) / 4
print(f'mean of place 1: {mean_even_index:.4f}')
print(f'mean of place 2: {mean_odd_index:.4f}')



end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
print(f"MODE = {MODE}")
print(f"Strict layer: {strict_linear}")
    

