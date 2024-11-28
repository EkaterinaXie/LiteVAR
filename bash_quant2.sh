#!/bin/bash
# 同时运行

# 0.95

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "4," --model_depth 30 --mode 0 --pic_num 50 --seed 0 --quant 1 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "4," --model_depth 30 --mode 1 --pic_num 50 --seed 0 --quant 1 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "3," --model_depth 30 --mode 2 --pic_num 50 --seed 0 --quant 1 --threshold 0.95 &


# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "6," --model_depth 30 --mode 0 --pic_num 50 --seed 0 --quant 2 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "6," --model_depth 30 --mode 1 --pic_num 50 --seed 0 --quant 2 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "5," --model_depth 30 --mode 2 --pic_num 50 --seed 0 --quant 2 --threshold 0.95 &


# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "5," --model_depth 30 --mode 0 --pic_num 50 --seed 0 --quant 3 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "1," --model_depth 30 --mode 1 --pic_num 50 --seed 0 --quant 3 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "1," --model_depth 30 --mode 2 --pic_num 50 --seed 0 --quant 3 --threshold 0.95 &



# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "2," --model_depth 30 --mode 0 --pic_num 50 --seed 0 --quant 4 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "2," --model_depth 30 --mode 1 --pic_num 50 --seed 0 --quant 4 --threshold 0.95 &

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "3," --model_depth 30 --mode 2 --pic_num 50 --seed 0 --quant 4 --threshold 0.95 &

python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "3," --model_depth 30 --mode 0 --pic_num 50 --seed 0 --quant 0 --threshold 0.95 &

# 查看生成的是否完全
# find /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/output_pic/seed_0 -type d -exec sh -c 'echo "{}"; find "{}" -type f | wc -l' sh {} \; > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/log_pic_num.txt