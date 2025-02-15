from torchvision.transforms import functional as F
from torchvision import transforms
import torch.nn.functional
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from torchmetrics.image.inception import InceptionScore
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchmetrics.multimodal.clip_score import CLIPScore
from diffusers.models.attention_processor import Attention
import time
import argparse

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

def evaluate_template_matching(order_list, cal_cost, pipe):
    count = {0: 0}
    for pattern_type in order_list:
        count[pattern_type] = 0
    mask_count_total = 0
    total = 0
    for name, module in pipe.transformer.named_modules():
        if isinstance(module, Attention):
            for timestep, mask_list in module.mask.items():
                pattern_list = np.zeros(16)
                for i in range(16):
                    type = 0
                    for j in order_list:
                        if mask_list[j][i]:
                            type = j
                            pattern_list[i] = j
                            break
                    count[type] += 1
                module.mask[timestep] = pattern_list

    total_num = sum(count.values())
    cal = 0
    for k, v in count.items():
        cal += cal_cost[k] * v / total_num
    print("template matching info: ")
    print(count)
    print("total percentage reduction: ", round(1 - cal, 2))


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (299, 299))


def save_output_hook(m, i, o):
    m.saved_output = o


def test_latencies(pipe, n_steps, calib_x, bs, only_transformer=True, test_attention=True):
    latencies = {}
    for b in bs:
        pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
        st = time.time()
        for i in range(3):
            pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
        ed = time.time()
        t = (ed - st) / 3
        if only_transformer:

            handler = pipe.transformer.register_forward_hook(save_output_hook)
            pipe([calib_x[0] for _ in range(b)], num_inference_steps=1)
            handler.remove()
            old_forward = pipe.transformer.forward
            pipe.transformer.forward = lambda *arg, **kwargs: pipe.transformer.saved_output
            st = time.time()
            for i in range(3):
                pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
            ed = time.time()
            t_other = (ed - st) / 3
            pipe.transformer.forward = old_forward
            del pipe.transformer.saved_output
            print(f"average time for other bs={b} inference: {t_other}")
            latencies[f"{b}_other"] = t_other
            latencies[f"{b}_transformer"] = t - t_other
        print(f"average time for bs={b} inference: {t}")
        latencies[f"{b}_all"] = t

        if test_attention:  # Test the latency of the attention modules
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention) and "attn1" in name:
                    module.old_forward = module.forward
                    module.forward = lambda *arg, **kwargs: arg[0]
            st = time.time()
            for i in range(3):
                pipe([calib_x[0] for _ in range(b)], num_inference_steps=n_steps)
            ed = time.time()
            t_other2 = (ed - st) / 3
            for name, module in pipe.transformer.named_modules():
                if isinstance(module, Attention) and "attn1" in name:
                    module.forward = module.old_forward
            t_attn = t - t_other2
            print(f"average time for attn bs={b} inference: {t_attn}")
            latencies[f"{b}_attn"] = t_attn
    return latencies


def evaluate_quantitative_scores(
    pipe,
    real_image_path,
    n_images=50000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images",
    guidance_scale=4,
):
    results = {}
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # Inception Score
    inception = InceptionScore().to(device)
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path):
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for i in range(0, n_images, batchsize):
        class_ids = np.random.randint(0, 1000, batchsize)
        output = pipe(
            class_labels=class_ids,
            generator=generator,
            output_type="np",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        fake_images = output.images
        # Inception Score
        torch_images = torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
        
        torch_images = torch.nn.functional.interpolate(
            torch_images, size=(299, 299), mode="bilinear", align_corners=False
        ).to(device)
        inception.update(torch_images) # 1 C 299 299 uint8

        for j, image in enumerate(fake_images):
            image = F.to_pil_image(image)
            image.save(f"{fake_image_path}/{i+j}.png")

    IS = inception.compute()
    results["IS"] = IS
    print(f"Inception Score: {IS}")

    fid_value = calculate_fid_given_paths(
        [real_image_path, fake_image_path],
        64,
        device,
        dims=2048,
        num_workers=8,
    )
    results["FID"] = fid_value
    print(f"FID: {fid_value}")
    return results


def evaluate_quantitative_scores_text2img(
    pipe,
    real_image_path,
    mscoco_anno,
    n_images=50000,
    batchsize=1,
    seed=3,
    num_inference_steps=20,
    fake_image_path="output/fake_images",
    reuse_generated=True,
    negative_prompt="",
    guidance_scale=4.5,
):
    results = {}
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # Inception Score
    inception = InceptionScore().to(device)
    clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path) and not reuse_generated:
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for index in range(0, n_images, batchsize):

        slice = mscoco_anno["annotations"][index : index + batchsize]
        filename_list = [str(d["id"]).zfill(12) for d in slice]
        print(f"Processing {index}th image")
        caption_list = [d["caption"] for d in slice]
        torch_images = []
        for filename in filename_list:
            image_file = f"{fake_image_path}/{filename}.jpg"
            if os.path.exists(image_file):
                image = Image.open(image_file)
                image_np = np.array(image)
                torch_image = torch.tensor(image_np).unsqueeze(0).permute(0, 3, 1, 2)
                torch_images.append(torch_image)
        if len(torch_images) > 0:
            torch_images = torch.cat(torch_images, dim=0)
            print(torch_images.shape)
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)
            inception.update(torch_images)
            clip.update(torch_images, caption_list[: len(torch_images)])
        else:
            output = pipe(
                caption_list,
                generator=generator,
                output_type="np",
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
            )
            fake_images = output.images
            # Inception Score
            count = 0
            torch_images = torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
            torch_images = torch.nn.functional.interpolate(
                torch_images, size=(299, 299), mode="bilinear", align_corners=False
            ).to(device)
            inception.update(torch_images)
            clip.update(torch_images, caption_list)
            for j, image in enumerate(fake_images):
                # image = image.astype(np.uint8)
                image = F.to_pil_image((image * 255).astype(np.uint8))
                image.save(f"{fake_image_path}/{filename_list[count]}.jpg")
                count += 1

    IS = inception.compute()
    CLIP = clip.compute()
    results["IS"] = IS
    results["CLIP"] = CLIP
    print(f"Inception Score: {IS}")
    print(f"CLIP Score: {CLIP}")

    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    fid_value = calculate_fid_given_paths(
        [real_image_path, fake_image_path],
        64,
        device,
        dims=2048,
        num_workers=8,
    )
    results["FID"] = fid_value
    print(f"FID: {fid_value}")
    return results


def method_speed_test(pipe):
    attn = pipe.transformer.transformer_blocks[0].attn1
    all_results = []
    for seqlen in [1024, 1024 * 4, 1024 * 16]:
        print(f"Test seqlen {seqlen}")
        for method in [
            "ori",
            "full_attn",
            "full_attn+cfg_attn_share",
            "residual_window_attn",
            "residual_window_attn+cfg_attn_share",
            "output_share",
        ]:
            if method == "ori":
                attn.set_processor(AttnProcessor2_0())
                attn.processor.need_compute_residual = [1]
                need_compute_residuals = [False]
            else:
                attn.set_processor(FastAttnProcessor([0, 0], [method]))
                if "full_attn" in method:
                    need_compute_residuals = [False, True]
                else:
                    need_compute_residuals = [False]
            for need_compute_residual in need_compute_residuals:
                attn.processor.need_compute_residual[0] = need_compute_residual
                # warmup
                x = torch.randn(2, seqlen, attn.query_dim).half().cuda()
                for i in range(10):
                    attn.stepi = 0
                    attn(x)
                torch.cuda.synchronize()
                st = time.time()
                for i in range(1000):
                    attn.stepi = 0
                    attn(x)
                torch.cuda.synchronize()
                et = time.time()
                print(f"Method {method} need_compute_residual {need_compute_residual} time {et-st}")
                all_results.append(et - st)
        print(all_results)

import json
def fid_scores_me(
    pipe,
    real_image_path,
    n_images=2000,
    batchsize=1,
    fake_image_path="output/fake_images",
):
    results = {}
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # Inception Score
    inception = InceptionScore().to(device)
    # FID
    np.random.seed(seed)
    generator = torch.manual_seed(seed)
    if os.path.exists(fake_image_path):
        os.system(f"rm -rf {fake_image_path}")
    os.makedirs(fake_image_path, exist_ok=True)
    for i in range(0, n_images, batchsize):
        # Inception Score
        # torch_images = torch.Tensor(fake_images * 255).byte().permute(0, 3, 1, 2).contiguous()
        if MODE == 0: # 原图
            # img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_original/demo_{MODEL_DEPTH}_original_{i}.png")
            img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_original/single_{MODEL_DEPTH}_original_{i}.png") # 0-1999
        elif MODE == 1: # 仅严格mask
            # img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_masked/demo_{MODEL_DEPTH}_masked_{i}.png")
            img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_masked/single_{MODEL_DEPTH}_masked_{i}.png")
        elif MODE == 2: # 全局reuse
            # img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/experiments/output_pic/seed_{seed}/{save_name}/all_cfg/demo_{MODEL_DEPTH}_masked_cfg_{i}.png")
            img = Image.open(f"/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/output_pic/single/seed_{seed}/{save_name}/all_cfg/single_{MODEL_DEPTH}_masked_cfg_{i}.png")
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

    fid_value = calculate_fid_given_paths(
        [real_image_path, fake_image_path],
        64,
        device,
        dims=2048,
        num_workers=8,
    )
    results["FID"] = fid_value
    print(f"FID: {fid_value}")
    return results