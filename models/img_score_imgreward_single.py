# image reward
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
from PIL import Image
import ImageReward as RM
import time
import json
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")

# 添加参数
parser.add_argument('--cuda_devices', type=str, default="4,", help='Specify the CUDA devices to use, e.g., "0,1" for using devices 0 and 1')
parser.add_argument('--model_depth', type=int, choices=[16, 20, 24, 30], required=True, help='Specify the model depth')
parser.add_argument('--mode', type=int, choices=[0, 1, 2, -1], required=True, help='Specify the mode: 0 for original, 1 for masked, 2 for global cfg, -1 for design mask')
parser.add_argument('--pic_num', type=int, default=250, help='Specify the number of images to generate')
parser.add_argument('--seed', type=int, default=0, help='Set the seed of the model')
parser.add_argument('--quant', type=int, default=0, help='no quant model') # quant=1:w8a8 
parser.add_argument('--try_num', type=int, default=1, help='use when run multi test at the same time')
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
try_num = args.try_num
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
print(f"strict layer: {args.strict}")

# 检查 CUDA 是否可用

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# 计算耗时
start_time = time.time()

# 加载 ImageNet 标签映射文件
with open('/share/public/diffusion_quant/xierui/pythonprogram/VAR/imagenet_class_index.json', 'r') as f:
    imagenet_labels = json.load(f)

class_labels_map = {}
label_name_all = {}

for i in range(PIC_NUM*8):
    label_name_group = []
    class_labels = i//8
    img_list = f'{class_labels}' 
    if img_list in imagenet_labels:
        class_label = imagenet_labels[img_list][1]
        label_name_group = class_label
    else:
        print(f"Label not found for class {img_list}")

    label_name_all[i] = label_name_group
# print(label_name_all)
'''quant'''
if QUANT == 0:
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
# if QUANT == 1:
#     save_name = f'quant/{strict_linear}/w8a8attn8'
# from transformers import AutoModelForSequenceClassification, AutoConfig

if __name__ == "__main__":

    model_path = '/home/xierui/.cache/ImageReward/ImageReward.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please download it now.")
        model = RM.load("ImageReward-v1.0").cuda()
    else:
        # 加载本地模型
        model = RM.load(model_path).cuda()
    # print(model.device) # cuda
    score = {}
    for i in range(PIC_NUM*8):
        print(f"IMAGE —— {i}")

        prompt = label_name_all[i]

        # 无quant mask来自多图
        img_list_1 = []  
        root_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/"
        if MODE == 0: # 原图
            img_list_1 = [os.path.join(root_dir, f"{save_name}/all_original/single_{MODEL_DEPTH}_original_{i}.png")]
        elif MODE == 1: # 仅严格mask
            img_list_1 = [os.path.join(root_dir, f"threshold_{threshold}/{save_name}/all_masked/single_{MODEL_DEPTH}_masked_{i}.png")]
        elif MODE == 2: # 全局reuse
            img_list_1 = [os.path.join(root_dir, f"threshold_{threshold}/{save_name}/all_cfg/single_{MODEL_DEPTH}_masked_cfg_{i}.png")]
        
        # caogao
        # img_list_1 = []  
        # root_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/"
        # if MODE == 0: # 原图
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_original/single_{MODEL_DEPTH}_original_{i}.png")
        #     img_list_1 = [os.path.join(root_dir, f"all_original/single_{MODEL_DEPTH}_original_{i}.png")]
        # elif MODE == 1: # 仅严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_masked/single_{MODEL_DEPTH}_masked_{i}.png")
        #     img_list_1 = [os.path.join(root_dir, f"all_masked/single_{MODEL_DEPTH}_masked_{i}.png")]
        # elif MODE == 2: # 全局reuse
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_cfg/single_{MODEL_DEPTH}_masked_cfg_{i}.png")
        #     img_list_1 = [os.path.join(root_dir, f"all_cfg/single_{MODEL_DEPTH}_masked_cfg_{i}.png")]
          
        with torch.no_grad():
            image_rewards = []
            ranking, rewards = model.inference_rank(prompt, img_list_1)
            image_rewards.append(rewards)
        score[i] = rewards
        print(image_rewards)
        print(f'SCORE OF EACH PICTURE: {score}')

    score_one_group = []
    score_one_group_mean = []
    # 初始化总和
    total_sum = 0
    # 将每个键对应列表的对应元素相加
    for key, value in score.items():
        total_sum += value
    print(f'sum IMAGE REWARD of all pics: {total_sum}')
    print(f'mean IMAGE REWARD of all pics: {total_sum/PIC_NUM}')


    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    print(f"MODE = {MODE}")
    print(f'seed = {seed}')
    print(f"quant = {QUANT}")

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "6," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 0 --try_num 0 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/caogao.txt