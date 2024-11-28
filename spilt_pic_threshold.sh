#!/bin/bash
echo "all original"
echo "MODE 0 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 0 --pic_num 1000 --seed 0 --quant 0
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "threshold 0.95"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.95
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo "MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.95
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "threshold 0.90"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.90
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo "MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.90
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "threshold 0.85"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.85
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo "MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.85
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "threshold 0.80"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.80
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo "MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.80
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "threshold 0.70"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.70
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo "MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.70
if [ $? -ne 0 ]; then
    echo "failed"
fi

echo "threshold 0.60"
echo "MODE 1 "
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 1 --pic_num 1000 --seed 0 --quant 0 --threshold 0.60
if [ $? -ne 0 ]; then
    echo "failed"
fi
echo "MODE 2"
python /share/public/diffusion_quant/xierui/pythonprogram/VAR/models/spilt_pic.py --cuda_devices "7," --model_depth 30 --mode 2 --pic_num 1000 --seed 0 --quant 0 --threshold 0.60
if [ $? -ne 0 ]; then
    echo "failed"
fi
