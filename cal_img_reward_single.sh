#!/bin/bash
# only no quant
# echo "all original"
# echo "MODE 0 "
# python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 0 --pic_num 1000 --seed 0 --quant 0 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/m0q0.txt
# if [ $? -ne 0 ]; then
#     echo "failed"
# fi


echo "threshold 0.95"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.95 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.95/m1q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo " MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.95 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.95/m2q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi


echo "threshold 0.90"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.90 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.9/m1q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo " MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.90 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.9/m2q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi


echo "threshold 0.85"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.85 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.85/m1q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo " MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.85 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.85/m2q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi


echo "threshold 0.80"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.80 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.8/m1q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo " MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.80 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.8/m2q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi


echo "threshold 0.70"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.70 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.7/m1q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo " MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.70 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.7/m2q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi


echo "threshold 0.60"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.60 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.6/m1q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo " MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/img_score_imgreward_single.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.60 > /share/public/diffusion_quant/xierui/pythonprogram/VAR/experiments_new/log/img_score_txt/seed_0/image_reward/threshold_0.6/m2q0.txt
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "finish all"