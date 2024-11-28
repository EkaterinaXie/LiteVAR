# 画一个距离直方图
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import torch
import matplotlib.pyplot as plt
level = [1, 2, 3, 4, 5, 6, 8, 10, 13, 16]
token_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256]
print('Now for depth = 16:\n')
pth_root = '/share/xierui-nfs/pythonProgram/VAR/attention16distnew/pth/'
visible_root = '/share/xierui-nfs/pythonProgram/VAR/attention16distnew/visible/'
distance_pic_path = '/share/xierui-nfs/pythonProgram/VAR/attention16distnew/pic/'
os.makedirs(pth_root, exist_ok=True)
os.makedirs(visible_root, exist_ok=True)
os.makedirs(distance_pic_path, exist_ok=True)
for scale in range(3, 10):
    # 创建保存图像的目录
    print(f'scale={scale}')
    # save_dir = f'/share/xierui-nfs/pythonProgram/VAR/attentionmap/attnpic30/hotmap_{scale}th/'
    # os.makedirs(save_dir, exist_ok=True)
    num_rows = 16
    num_cols = 16
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 6 * num_rows))
    if scale == 3: # 两个mask
    # 加载保存的 .pth
        distances1 = torch.load(pth_root + f'scale{scale}_dis1.pth')
        distances2 = torch.load(pth_root + f'scale{scale}_dis2.pth')
        visible1 = torch.load(visible_root + f'scale{scale}_visible1.pth')
        visible2 = torch.load(visible_root + f'scale{scale}_visible2.pth')
        print(f'length of distances1 = {len(distances1)}')
        print(f'length of distances2 = {len(distances2)}')
        for k in range(num_rows * num_cols):
            distance1 = distances1[k]
            visible_num1 = visible1[k]
            distance1 = distance1.cpu().numpy()
            visible_num1 = visible_num1.cpu().numpy()
            sorted_distances = np.sort(distance1)
            # sorted_distances = distance1.sort().values  # 对距离进行排序
            sorted_distances = sorted_distances[:visible_num1]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances1.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        for k in range(num_rows * num_cols):
            distance2 = distances2[k]
            visible_num2 = visible2[k]
            distance2 = distance2.cpu().numpy()
            visible_num2 = visible_num2.cpu().numpy()
            # sorted_distances = distance2.sort().values  # 对距离进行排序
            sorted_distances = np.sort(distance2)
            sorted_distances = sorted_distances[:visible_num2]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances2.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
    elif scale == 4 or scale == 5: # 3个mask
        distances1 = torch.load(pth_root + f'scale{scale}_dis1.pth')
        distances2 = torch.load(pth_root + f'scale{scale}_dis2.pth')
        distances3 = torch.load(pth_root + f'scale{scale}_dis3.pth')
        visible1 = torch.load(visible_root + f'scale{scale}_visible1.pth')
        visible2 = torch.load(visible_root + f'scale{scale}_visible2.pth')
        visible3 = torch.load(visible_root + f'scale{scale}_visible3.pth')
        print(f'length of distances1 = {len(distances1)}')
        print(f'length of distances2 = {len(distances2)}')
        print(f'length of distances3 = {len(distances3)}')
        for k in range(num_rows * num_cols):
            distance1 = distances1[k]
            visible_num1 = visible1[k]
            distance1 = distance1.cpu().numpy()
            visible_num1 = visible_num1.cpu().numpy()
            sorted_distances = np.sort(distance1)
            # sorted_distances = distance1.sort().values  # 对距离进行排序
            sorted_distances = sorted_distances[:visible_num1]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)    
        save_path = os.path.join(path, f'all_distances1.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        for k in range(num_rows * num_cols):
            distance2 = distances2[k]
            visible_num2 = visible2[k]
            distance2 = distance2.cpu().numpy()
            visible_num2 = visible_num2.cpu().numpy()
            # sorted_distances = distance2.sort().values  # 对距离进行排序
            sorted_distances = np.sort(distance2)
            sorted_distances = sorted_distances[:visible_num2]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances2.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        for k in range(num_rows * num_cols):
            distance3 = distances3[k]
            visible_num3 = visible3[k]
            distance3 = distance3.cpu().numpy()
            visible_num3 = visible_num3.cpu().numpy()
            # sorted_distances = distance2.sort().values  # 对距离进行排序
            sorted_distances = np.sort(distance3)
            sorted_distances = sorted_distances[:visible_num3]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances3.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
    elif scale == 6 or scale == 7 or scale == 8: # 4个mask
        distances1 = torch.load(pth_root + f'scale{scale}_dis1.pth')
        distances2 = torch.load(pth_root + f'scale{scale}_dis2.pth')
        distances3 = torch.load(pth_root + f'scale{scale}_dis3.pth')
        distances4 = torch.load(pth_root + f'scale{scale}_dis4.pth')
        visible1 = torch.load(visible_root + f'scale{scale}_visible1.pth')
        visible2 = torch.load(visible_root + f'scale{scale}_visible2.pth')
        visible3 = torch.load(visible_root + f'scale{scale}_visible3.pth')
        visible4 = torch.load(visible_root + f'scale{scale}_visible4.pth')
        print(f'length of distances1 = {len(distances1)}')
        print(f'length of distances2 = {len(distances2)}')
        print(f'length of distances3 = {len(distances3)}')
        print(f'length of distances4 = {len(distances4)}')
        for k in range(num_rows * num_cols):
            distance1 = distances1[k]
            visible_num1 = visible1[k]
            distance1 = distance1.cpu().numpy()
            visible_num1 = visible_num1.cpu().numpy()
            sorted_distances = np.sort(distance1)
            # sorted_distances = distance1.sort().values  # 对距离进行排序
            sorted_distances = sorted_distances[:visible_num1]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances1.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        for k in range(num_rows * num_cols):
            distance2 = distances2[k]
            visible_num2 = visible2[k]
            distance2 = distance2.cpu().numpy()
            visible_num2 = visible_num2.cpu().numpy()
            # sorted_distances = distance2.sort().values  # 对距离进行排序
            sorted_distances = np.sort(distance2)
            sorted_distances = sorted_distances[:visible_num2]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances2.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        for k in range(num_rows * num_cols):
            distance3 = distances3[k]
            visible_num3 = visible3[k]
            distance3 = distance3.cpu().numpy()
            visible_num3 = visible_num3.cpu().numpy()
            # sorted_distances = distance2.sort().values  # 对距离进行排序
            sorted_distances = np.sort(distance3)
            sorted_distances = sorted_distances[:visible_num3]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances3.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        for k in range(num_rows * num_cols):
            distance4 = distances4[k]
            visible_num4 = visible4[k]
            distance4 = distance4.cpu().numpy()
            visible_num4 = visible_num4.cpu().numpy()
            # sorted_distances = distance2.sort().values  # 对距离进行排序
            sorted_distances = np.sort(distance4)
            sorted_distances = sorted_distances[:visible_num4]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances4.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
    elif scale == 9: # 1个mask
        distances1 = torch.load(pth_root + f'scale{scale}_dis1.pth')
        visible1 = torch.load(visible_root + f'scale{scale}_visible1.pth')
        
        print(f'length of distances1 = {len(distances1)}')
        
        for k in range(num_rows * num_cols):
            distance1 = distances1[k]
            visible_num1 = visible1[k]
            distance1 = distance1.cpu().numpy()
            visible_num1 = visible_num1.cpu().numpy()
            sorted_distances = np.sort(distance1)
            # sorted_distances = distance1.sort().values  # 对距离进行排序
            sorted_distances = sorted_distances[:visible_num1]  # 只保留前 visible_num1 个数值
            unique_distances, counts = np.unique(sorted_distances, return_counts=True)
            total_count = counts.sum()
            proportions = counts / total_count
            # 计算当前子图的索引
            row_index = k // num_cols
            col_index = k % num_cols
            # 绘制直方图
            axs[row_index, col_index].bar(unique_distances, proportions)
            axs[row_index, col_index].set_xlabel('Distance')
            axs[row_index, col_index].set_ylabel('Proportion')
            axs[row_index, col_index].set_title(f'Distance Histogram {k}')
        plt.tight_layout()  # 调整子图之间的间距
        # 指定保存图像的完整文件路径
        path = os.path.join(distance_pic_path, f'scale_{scale}/')
        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, f'all_distances1.png')
        plt.savefig(save_path)
        plt.close()
        plt.clf()
        
            
    