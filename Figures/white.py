import cv2
import numpy as np

# 加载图像
image = cv2.imread('/mnt/public/diffusion_quant/xierui/pythonprogram/VAR/Figures/attention_map_4.png')

# 计算RGB平均值
avg_rgb = np.mean(image, axis=2)

# 设置阈值
threshold = 200  # 根据需要调整

# 创建掩膜
mask = avg_rgb > threshold

# 替换大于阈值的部分为白色
image[mask] = [255, 255, 255]

# 保存结果
cv2.imwrite('output_image.jpg', image)