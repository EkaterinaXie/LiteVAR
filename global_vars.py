import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,"
import os.path as osp
import torch
import torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import matplotlib.pyplot as plt
import time
import tkinter as tk
MODEL_DEPTH = 30    # TODO: =====> please specify MODEL_DEPTH <=====
assert MODEL_DEPTH in {16, 20, 24, 30}
# PIC_NUM = 10
seed = 0
MODE = 0        # 0:原图 
                # 1:仅严格scale不合并cfg 
                # 2:全局cfg
                # -1:设计mask的模式，上述三个是inference模式