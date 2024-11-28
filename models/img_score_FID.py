from torchvision.transforms import functional as F
from torchvision import transforms
import torch.nn.functional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
# from torchmetrics.multimodal.clip_score import CLIPScore
from diffusers.models.attention_processor import Attention
import time
import argparse
import time
import shutil
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset

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
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = [os.path.join(root, f) for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image
    
#######################################
n_images = 50000
batchsize = 1

results = {}
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# Inception Score
inception = InceptionScore().to(device)
# FID
np.random.seed(seed)
generator = torch.manual_seed(seed)
real_image_path = "/mnt/public/yuanzhihang/imagenet/ILSVRC/Data/CLS-LOC/val/"
# real_image_files = os.listdir(real_image_path)
# real_image_subset = random.sample(real_image_files, PIC_NUM)
real_image_subset_path = "/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/real2000_from_imgnet/"
os.makedirs(real_image_subset_path, exist_ok=True)
# 将选择的图像复制到临时文件夹
# for file_name in real_image_subset:
#     src = os.path.join(real_image_path, file_name)
#     dst = os.path.join(real_image_subset_path, file_name)
#     shutil.copy(src, dst)

# if MODE == 0: # 原图
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_original/"

# elif MODE == 1: # 仅严格mask
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_masked/"

# elif MODE == 2: # 全局reuse
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_cfg/"
# yiban
# if MODE == 0: # 原图
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/{save_name}/all_original/"

# elif MODE == 1: # 仅严格mask
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/{save_name}/all_masked/"

# elif MODE == 2: # 全局reuse
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/{save_name}/all_cfg/"

# mix-percision
# if MODE == 0:
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/mix-percision/{save_name}/fc2-16/all_original/"
# elif MODE == 1: # 仅严格mask
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_masked/"

# elif MODE == 2: # 全局reuse
#     fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/threshold_{threshold}/mix-percision/{save_name}/fc2-16/all_cfg/"
# strict layer
if MODE == 0:
    # fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/single/seed_{seed}/{save_name}/all_original/"
    fake_image_path = f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_original/" 

# 计算耗时
start_time = time.time()

for i in range(0, PIC_NUM*8):
    # Inception Score
    # torch_images = torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
    if MODE == 0: # 原图
        img_name = f"single_{MODEL_DEPTH}_original_{i}.png" # 0-1999
    elif MODE == 1: # 仅严格mask
        img_name = f"single_{MODEL_DEPTH}_masked_{i}.png" # 0-1999
    elif MODE == 2: # 全局reuse
        img_name = f"single_{MODEL_DEPTH}_cfg_{i}.png" # 0-1999
    img = Image.open(fake_image_path+img_name)
    fake_img_array = np.array(img)
    fake_img_tensor = torch.from_numpy(fake_img_array).unsqueeze(0)
    fake_img_tensor = fake_img_tensor.permute(0, 3, 1, 2) # 1,3,256,256
    torch_images = torch.nn.functional.interpolate(
        fake_img_tensor, size=(299, 299), mode="bilinear", align_corners=False
    ).to(device) # 1 3 299 299
    inception.update(torch_images) # 1 C 299 299 uint8

IS = inception.compute()
results["IS"] = IS
print(f"Inception Score: {IS}")

# # 设置图像预处理转换
# transform = transforms.Compose([
#     transforms.Resize((299, 299)),  # Inception 网络接受的输入尺寸
#     transforms.ToTensor(),          # 转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
# ])

# # 创建图像数据集
# real_dataset = CustomDataset(root=real_image_path, transform=transform)
# fake_dataset = CustomDataset(root=fake_image_path, transform=transform)

# # 创建数据加载器
# real_loader = DataLoader(real_dataset, batch_size=64, shuffle=False, num_workers=8)
# fake_loader = DataLoader(fake_dataset, batch_size=64, shuffle=False, num_workers=8)
def resize_images(image_dir, new_size, output_dir):
    """
    Resizes all images in the given directory to the specified size.
    
    Args:
        image_dir (str): Path to the directory containing images.
        new_size (tuple): The new size of the images (width, height).
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    img = img.resize(new_size, resample=Image.BICUBIC)
                    output_path = os.path.join(output_dir, filename)
                    img.save(output_path)
            except Exception as e:
                print(f"Error resizing {file_path}: {e}")

# Specify the desired new size for the images
new_image_size = (299,299)  # Example size (width, height)
resized_real_img_path = "/mnt/public/yuanzhihang/imagenet/ILSVRC/Data/CLS-LOC/val_299x299/"
# resize_images(real_image_path, new_image_size, resized_real_img_path)
# resize_images(fake_image_path, new_image_size)
fid_value = calculate_fid_given_paths(
    [resized_real_img_path, fake_image_path],
    # [real_image_subset_path, fake_image_path],
    64,
    device,
    dims=2048,
    num_workers=8,
)
# fid_value = calculate_fid_given_paths(
#     [real_loader, fake_loader],
#     device,
#     dims=2048,
#     num_workers=8
# )
end_time = time.time()
execution_time = end_time - start_time
results["FID"] = fid_value
print(f"FID: {fid_value}")

print("Execution time:", execution_time, "seconds")