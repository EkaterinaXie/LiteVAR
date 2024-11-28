# image reward
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
from PIL import Image
import ImageReward as RM
import time
import json
import argparse
# MODEL_DEPTH = 30
# PIC_NUM = 250
# MODE = 1 # 0:原图 1:仅严格mask 2:cfg的全局reuse
# 创建解析器
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
# 初始化一个空的 class_labels_map 字典
class_labels_map = {}
label_name_all = {}

for i in range(PIC_NUM):
    # print(f"NOW IMAGE —— {i}")
    label_name_group = []
    # class_labels_map[i] = tuple([item for j in range(4) for item in (j + 4 * i, j + 4 * i)])
    # {0: (0, 0, 1, 1, 2, 2, 3, 3), 1: (4, 4, 5, 5, 6, 6, 7, 7),...249: (996, 996, 997, 997, 998, 998, 999, 999)}
    class_labels_map[i] = tuple([item for j in range(4) for item in (i, i)])

    class_labels = class_labels_map[i]

    for class_index in class_labels:
        img_list = f'{class_index}' 
        # 查找类别对应的标签
        if img_list in imagenet_labels:
            class_label = imagenet_labels[img_list][1]
            # print(f"Class {img_list} label: {class_label}")
            label_name_group.append(class_label)
        else:
            print(f"Label not found for class {img_list}")

    # print(label_name_group)
    label_name_all[i] = label_name_group
# print(label_name_all)
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
if QUANT == 1:
    save_name = f'quant/{strict_linear}/w8a8attn8'
from transformers import AutoModelForSequenceClassification, AutoConfig

if __name__ == "__main__":

    model_path = '/home/xierui/.cache/ImageReward/ImageReward.pt'
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please download it now.")
        model = RM.load("ImageReward-v1.0").cuda()
        # model = AutoModelForSequenceClassification.from_pretrained("ImageReward-v1.0")
    else:
        # 加载本地模型
        model = RM.load(model_path).cuda()
    # print(model.device) # cuda
    score = {}
    for i in range(PIC_NUM):
        print(f"IMAGE —— {i}")

        # prompt = []
        prompt = [None] * 8  # Initialize prompt list with None values
        for index in range(8):
            prompt[index] = label_name_all[i][index]    

        # 打开 PNG 图像
        # if MODE == 0: # 原图
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
        # elif MODE == 1: # 仅严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
        # elif MODE == 2: # 全局reuse
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
            
        # # group
        # if MODE == 0: # 原图
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_original/d_{MODEL_DEPTH}_original_{i}.png")
        # elif MODE == 1: # 仅严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_masked/d_{MODEL_DEPTH}_masked_{i}.png")
        # elif MODE == 2: # 全局reuse
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/{save_name}/all_cfg/d_{MODEL_DEPTH}_masked_cfg_{i}.png")
        
        # # a: per token w: per channel
        # if MODE == 0: # 原图
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
        # elif MODE == 1: # 仅严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
        # elif MODE == 2: # 全局reuse
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
        # # 无quant mask来自多图
        # if MODE == 0: # 原图
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
        # elif MODE == 1: # 仅严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
        # elif MODE == 2: # 全局reuse
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
        # # mix percision
        # if MODE == 0: # MODE = 0 原图，无任何操作
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
        # elif MODE == 1: # MODE = 1 不合并cond和uncond，仅全scale严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
        # elif MODE == 2: # 目前采取全局cfg合并
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
        # strict layer
        # /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0/quant/ada_lin.1/w8a8attn8/all_original/demo_30_original_0.png
        if MODE == 0: # MODE = 0 原图，无任何操作
            img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
        # elif MODE == 1: # MODE = 1 不合并cond和uncond，仅全scale严格mask
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
        # elif MODE == 2: # 目前采取全局cfg合并
        #     img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
        
        # 获取图像的宽度和高度
        width, height = img.size

        # 定义新图像的高度（与原图像相同），每一列的宽度
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
            column_img = img.crop((left, upper, right, lower))
            temp_dir = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/seed_{seed}/temp/try{try_num}/{MODE}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png"
            os.makedirs(temp_dir, exist_ok=True)
            # column_img.save(temp_dir+f'column_{pic}.png')  # 保存每一列为单独的图像
            column_img.save(f"{temp_dir}column_{pic}.png")  # 保存每一列为单独的图像
            img_list_1.append([column_img])
        with torch.no_grad():
            image_rewards = []
            for index in range(8):
                ranking, rewards = model.inference_rank(prompt[index], img_list_1[index])
                image_rewards.append(rewards)
        score[i] = image_rewards
        print(image_rewards)
        # print(f'SCORE OF EACH PICTURE: {score}')

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
    for index in range(len(score_one_group)):
        sum_each_place = score_one_group[index]
        # print(f"sum IMAGE REWARD of the {index} place: {sum_each_place:.4f}")
    print(f'sum IMAGE REWARD of all pics: {score_one_group}')
    # 将列表中的每个值都除以给定的数
    score_one_group_mean = [num / PIC_NUM for num in score_one_group]

    for index in range(len(score_one_group_mean)):
        mean_each_place = score_one_group_mean[index]
        mean_of_pic_0 = score_one_group_mean[index]

        # print(f"mean IMAGE REWARD of the {index} place: {mean_each_place:.4f}")

    # print(f'mean IMAGE REWARD of all pictures: {score_one_group_mean:.4f}')
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
    print(f'seed = {seed}')
    print(f"quant = {QUANT}")
