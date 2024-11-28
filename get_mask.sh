#!/bin/bash

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "3," --model_depth 30 --mode -1 --pic_num 10 --seed 0 --quant 0 --threshold 0.60 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/30/threshold_0.60.txt

# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "4," --model_depth 30 --mode -1 --pic_num 10 --seed 0 --quant 0 --threshold 0.70 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/30/threshold_0.70.txt
echo "################# theshold 0.80 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "7," --model_depth 30 --mode -1 --pic_num 10 --seed 0 --quant 0 --threshold 0.80 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/30/threshold_0.80.txt
if [ $? -ne 0 ]; then
    echo "theshold 0.80 failed"
fi

echo "################# theshold 0.85"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "7," --model_depth 30 --mode -1 --pic_num 10 --seed 0 --quant 0 --threshold 0.85 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/30/threshold_0.85.txt
if [ $? -ne 0 ]; then
    echo "theshold 0.85 failed"
fi

echo "################# theshold 0.90"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/demo_sample_batch_quant.py --cuda_devices "7," --model_depth 30 --mode -1 --pic_num 10 --seed 0 --quant 0 --threshold 0.90 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/save_window/30/threshold_0.90.txt
if [ $? -ne 0 ]; then
    echo "theshold 0.90 failed"
fi
