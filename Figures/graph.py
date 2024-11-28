import matplotlib.pyplot as plt
import numpy as np

# 层名列表
layer_names = [
    'word_embed',
    'attn.mat_qkv',
    'attn.proj',
    'ffn.fc1',
    'ffn.fc2',
    'ada_lin.1',
    'head'
]

ir_std = -0.2791
is_std = 118.3725
fid_std = 24.4832

# 假设量化前后各项指标的数据
image_reward_diffs = [ir_std+0.2796, ir_std+0.3438, ir_std+0.3400, ir_std+0.2960, ir_std+1.3359, ir_std+0.3059, ir_std+0.3450]
is_diffs = [is_std-117.5732, is_std-111.5886, is_std-114.2654, is_std-116.1325, is_std-34.3094, is_std-115.3622, is_std-115.4266]
fid_diffs = [25.2494-24.4832, 24.5848-24.4832, 24.7486-24.4832, 25.1047-24.4832, 54.5430-24.4832, 24.9981-24.4832, 24.8638-24.4832]


image_reward_diffs = [x / (ir_std + 1.3359) for x in image_reward_diffs]
is_diffs = [x / (is_std - 34.3094) for x in is_diffs]
fid_diffs = [x / (54.5430-24.4832) for x in fid_diffs]

# 设置柱状图的宽度
bar_width = 0.25
x = np.arange(len(layer_names))

# 创建合并柱状图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制各项指标的柱状图
ax.bar(x - bar_width, image_reward_diffs, width=bar_width, label='Image Reward Differences', color='blue')
ax.bar(x, is_diffs, width=bar_width, label='IS Differences', color='orange')
ax.bar(x + bar_width, fid_diffs, width=bar_width, label='FID Differences', color='green')

# 设置图表标题和坐标轴标签
ax.set_title('Relative Differences in Metrics by Layer')
ax.set_xlabel('Layer Names')
ax.set_ylabel('Relative Differences')
ax.set_xticks(x)
ax.set_xticklabels(layer_names, rotation=45, ha='right')

# 添加图例
ax.legend()

# 自动调整子图参数, 使之填充整个图像区域
plt.tight_layout()

# 保存图表
plt.savefig('combined_bar_chart.png')

# 显示图表
plt.show()