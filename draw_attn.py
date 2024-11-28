import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
import matplotlib.pyplot as plt

# 遍历 scale 的范围从 0 到 9
print('Now for depth = 16:\n')
for scale in range(10):
    # 创建保存图像的目录
    print(f'scale={scale}')
    save_dir = f'/share/xierui-nfs/pythonProgram/VAR/attentionmap/attnpic16/hotmap_{scale}th/'
    os.makedirs(save_dir, exist_ok=True)

    # 加载保存的 .pth 文件
    attention_weights = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention16/attentionmap{scale}.pth')

    # 将 CUDA 张量复制到主机内存上
    attention_weights_cpu = attention_weights.cpu()
    print(attention_weights_cpu.shape)

    # 遍历 attention_weights_cpu 中的所有元素
    for i in range(attention_weights_cpu.shape[0]):
        for j in range(attention_weights_cpu.shape[1]):
            # 提取对应位置的二维矩阵
            data = attention_weights_cpu[i, j]
            print(data)
            # # 调整颜色映射的范围
            # vmin = data.min()
            # vmax = data.max()

            # 绘制注意力热力图
            plt.imshow(data, cmap='hot')
            plt.colorbar()

            # 刷新图形
            plt.gcf().canvas.draw()

            # 保存图像
            filename = f'{save_dir}[{i},{j}].png'
            plt.savefig(filename)
            # 清除当前图形
            plt.clf()

print('Now for depth = 30:\n')
for scale in range(10):
    # 创建保存图像的目录
    print(f'scale={scale}')
    save_dir = f'/share/xierui-nfs/pythonProgram/VAR/attentionmap/attnpic30/hotmap_{scale}th/'
    os.makedirs(save_dir, exist_ok=True)

    # 加载保存的 .pth 文件
    attention_weights = torch.load(f'/share/xierui-nfs/pythonProgram/VAR/attention30/attentionmap{scale}.pth')

    # 将 CUDA 张量复制到主机内存上
    attention_weights_cpu = attention_weights.cpu()
    print(attention_weights_cpu.shape)

    # # 遍历 attention_weights_cpu 中的所有元素
    # for i in range(attention_weights_cpu.shape[0]):
    #     for j in range(attention_weights_cpu.shape[1]):
    #         # 提取对应位置的二维矩阵
    #         data = attention_weights_cpu[i, j]

    #         # # 调整颜色映射的范围
    #         # vmin = data.min()
    #         # vmax = data.max()

    #         # 绘制注意力热力图
    #         plt.imshow(data, cmap='hot')
    #         plt.colorbar()

    #         # 刷新图形
    #         plt.gcf().canvas.draw()

    #         # 保存图像
    #         filename = f'{save_dir}[{i},{j}].png'
    #         plt.savefig(filename)
    #         # 清除当前图形
    #         plt.clf()