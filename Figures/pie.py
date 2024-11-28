import matplotlib.pyplot as plt
import numpy as np

# 设置较亮的莫兰蒂色系配色
morandi_colors = ['#D5CABD', '#A9C3A1', '#D0B3E4', '#A5D8D1', '#D4A5A7', '#E0B89F', '#E0CEC7', '#B5C0C4']

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
image_reward_diffs = [ir_std + 0.2796, ir_std + 0.3438, ir_std + 0.3400, ir_std + 0.2960, ir_std + 1.3359, ir_std + 0.3059, ir_std + 0.3450]
is_diffs = [is_std - 117.5732, is_std - 111.5886, is_std - 114.2654, is_std - 116.1325, is_std - 34.3094, is_std - 115.3622, is_std - 115.4266]
fid_diffs = [25.2494 - 24.4832, 24.5848 - 24.4832, 24.7486 - 24.4832, 25.1047 - 24.4832, 54.5430 - 24.4832, 24.9981 - 24.4832, 24.8638 - 24.4832]

# 归一化处理
image_reward_diffs = [x / (ir_std + 1.3359) for x in image_reward_diffs]
is_diffs = [x / (is_std - 34.3094) for x in is_diffs]
fid_diffs = [x / (54.5430 - 24.4832) for x in fid_diffs]

# 饼状图数据，包含所有层
data_sets = [
    image_reward_diffs,
    is_diffs,
    fid_diffs
]
titles = ['Image Reward', 'IS', 'FID']

# 自定义函数，只显示 ffn.fc2 的数值
def autopct_func(pct, allvalues):
    absolute = int(np.round(pct / 100. * np.sum(allvalues)))
    if pct > 0 and absolute == int(np.max(allvalues)):  # 只显示 ffn.fc2 的百分比
        return f'{pct:.1f}%'
    else:
        return ''

# 创建饼状图
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 分别绘制3个饼状图
for i, ax in enumerate(axs):
    wedges, texts, autotexts = ax.pie(
        data_sets[i], 
        autopct=lambda pct: autopct_func(pct, data_sets[i]),  # 只对最大值（ffn.fc2）显示数值
        startangle=90, 
        colors=morandi_colors
    )
    ax.set_title(titles[i])
    
fig.suptitle('Quantization Impact on Metrics by Layer', fontsize=16)

# 创建单独的图例
fig.legend(layer_names, loc="center right", fontsize='large')

# 调整布局
plt.tight_layout(rect=[0, 0, 0.85, 1])  # 留出右侧位置放置图例

# 保存图表
plt.savefig('individual_layer_pie_charts_ffn_fc2_only.png')