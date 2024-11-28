#!/bin/bash

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "0," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "word_embed" &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "1," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "attn.mat_qkv" &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "2," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "attn.proj" &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "3," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "ffn.fc1" &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "4," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "ffn.fc2" &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "5," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "ada_lin.1" &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "1," --model_depth 30 --mode 0 --pic_num 20 --seed 0 --quant 1 --strict "head" &
