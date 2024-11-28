# 打最后一个mask大对角，scale从3到9
# 最后一个mask：3、4、5、6、7、8、9
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
print('now for model depth = 16:')
rows = 16
cols = 16
for scale in range(3,10):
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention16/attentionmap{scale}.pth')
    # 创建一个 16x16 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
    edge = token_scale[scale]
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, -edge:].shape}')
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, -edge:].cpu()
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor - y_cor) <= win_size
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention16/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask1.png')
    # 保存绘图
    plt.savefig(save_dir)

    # # 显示大图
    # plt.show()


print('now for model depth = 30:')
rows = 16
cols = 30
for scale in range(3,10):
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention30/attentionmap{scale}.pth')
    # 创建一个 16x30 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(60, 32))
    edge = token_scale[scale]
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, -edge:].shape}')
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, -edge:].cpu()
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor - y_cor) <= win_size
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention30/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask1.png')
    # 保存绘图
    plt.savefig(save_dir)


# 打倒数第二个mask对角，scale从3到9 # 0，1，2不需要mask/3两个mask/4，5三个mask/6，7，8四个mask/9一个mask
# 倒数第二个mask：3、4、5、6、7、8
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
print('now for model depth = 16:')
rows = 16
cols = 16
for scale in range(3,9): #################
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention16/attentionmap{scale}.pth')
    # 创建一个 16x16 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
    edge = token_scale[scale]
    left_edge = tokenall_scale[scale-2] #################
    right_edge = tokenall_scale[scale-1] #####################
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, left_edge:right_edge].shape}') #######################
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, left_edge:right_edge].cpu() ##############
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor*(right_edge-left_edge)/edge - y_cor) <= win_size #################
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention16/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask2.png') ######################
    # 保存绘图
    plt.savefig(save_dir)

    # # 显示大图
    # plt.show()


print('now for model depth = 30:')
rows = 16
cols = 30
for scale in range(3,9):
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention30/attentionmap{scale}.pth')
    # 创建一个 16x30 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(60, 32))
    edge = token_scale[scale]
    left_edge = tokenall_scale[scale-2] #################
    right_edge = tokenall_scale[scale-1] #####################
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, left_edge:right_edge].shape}') #######################
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, left_edge:right_edge].cpu() ##############
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor*(right_edge-left_edge)/edge - y_cor) <= win_size #################
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention30/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask2.png') ######################
    # 保存绘图
    plt.savefig(save_dir)


# 倒数第三个mask：4、5、6、7、8
import matplotlib.pyplot as plt
import numpy as np
import os
level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
print('now for model depth = 16:')
rows = 16
cols = 16
for scale in range(4,9): #################
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention16/attentionmap{scale}.pth')
    # 创建一个 16x16 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
    edge = token_scale[scale]
    left_edge = tokenall_scale[scale-3] #################
    right_edge = tokenall_scale[scale-2] #####################
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, left_edge:right_edge].shape}') #######################
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, left_edge:right_edge].cpu() ##############
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor*(right_edge-left_edge)/edge - y_cor) <= win_size #################
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention16/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask3.png') ######################
    # 保存绘图
    plt.savefig(save_dir)

    # # 显示大图
    # plt.show()


print('now for model depth = 30:')
rows = 16
cols = 30
for scale in range(4,9): ########################
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention30/attentionmap{scale}.pth')
    # 创建一个 16x30 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(60, 32))
    edge = token_scale[scale]
    left_edge = tokenall_scale[scale-3]  #################
    right_edge = tokenall_scale[scale-2] #####################
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, left_edge:right_edge].shape}') #######################
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, left_edge:right_edge].cpu() ##############
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor*(right_edge-left_edge)/edge - y_cor) <= win_size #################
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention30/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask3.png') ######################
    # 保存绘图
    plt.savefig(save_dir)

# 倒数第四个mask：6、7、8
import matplotlib.pyplot as plt
import numpy as np
import os
level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
tokenall_scale = [1, 5, 14, 30, 55, 91, 155, 255, 424, 680]
print('now for model depth = 16:')
rows = 16
cols = 16
for scale in range(6,9): #################
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention16/attentionmap{scale}.pth')
    # 创建一个 16x16 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(32, 32))
    edge = token_scale[scale]
    left_edge = tokenall_scale[scale-4] #################
    right_edge = tokenall_scale[scale-3] #####################
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, left_edge:right_edge].shape}') #######################
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, left_edge:right_edge].cpu() ##############
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor*(right_edge-left_edge)/edge - y_cor) <= win_size #################
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention16/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask4.png') ######################
    # 保存绘图
    plt.savefig(save_dir)

    # # 显示大图
    # plt.show()


print('now for model depth = 30:')
rows = 16
cols = 30
for scale in range(6,9): ########################
    attentionmap = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention30/attentionmap{scale}.pth')
    # 创建一个 16x30 的大图，将 256 个子图放置在其中
    fig, axes = plt.subplots(rows, cols, figsize=(60, 32))
    edge = token_scale[scale]
    left_edge = tokenall_scale[scale-4] #################
    right_edge = tokenall_scale[scale-3] #####################
    print(f'now scale is {scale}')
    print(f'mask is {attentionmap[0, 0, :, left_edge:right_edge].shape}') #######################
    for i in range(rows):
        for j in range(cols):
            # 提取 attentionmap 的子图
            y = attentionmap[i, j, :, left_edge:right_edge].cpu() ##############
            y_np = y.detach().cpu().numpy()
            
            # 计算 attn_fractions
            win_sizes = np.arange(64)
            attn_fractions = []
            for win_size in win_sizes:
                x_cor, y_cor = torch.meshgrid(torch.arange(y.shape[0]), torch.arange(y.shape[1]))
                mask = torch.abs(x_cor*(right_edge-left_edge)/edge - y_cor) <= win_size #################
                attn_fraction = (mask * y).sum() / y.sum()
                attn_fractions.append(attn_fraction)
            
            # 在对应的子图中绘制数据
            ax = axes[i, j]
            ax.plot(win_sizes, attn_fractions, 'o', markersize=1)
            ax.set_title(f'Attn[{i}][{j}]')
            ax.set_xlim([0, 64])  # 设置 x 轴的范围
            ax.set_ylim([0, 1])  # 设置 y 轴的范围
            # 在纵坐标上标记特定位置
            ax.set_yticks([0.5, 0.85, 0.9, 0.95])
            # 在横坐标上标记特定位置
            ax.set_xticks([16, 32, 48, 64])

            # 找到超过 0.9 的第一个点
            for index, fraction in enumerate(attn_fractions):
                if fraction > 0.95:
                    ax.plot(win_sizes[index], fraction, 'ro', markersize=1)  # 绘制红色点
                    ax.text(win_sizes[index], fraction, str(win_sizes[index]), color='red', fontsize=8, ha='center', va='bottom')  # 标记横坐标
                    ax.plot([win_sizes[index], win_sizes[index]], [0, fraction], 'r--', markersize=0.3)  # 连接红色点到y轴
                    ax.plot([0, win_sizes[index]], [fraction, fraction], 'r--', markersize=0.3)  # 连接红色点到x轴
                    break
    # 调整子图之间的间距和边界
    plt.tight_layout()
    save_root = '/share/xierui-nfs/pythonProgram/VAR/attention30/windowsize/'
    os.makedirs(save_root, exist_ok=True)
    save_dir = os.path.join(save_root, f'scale{scale}_mask4.png') ######################
    # 保存绘图
    plt.savefig(save_dir)