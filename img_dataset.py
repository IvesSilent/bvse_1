# -* coding=utf8 *-
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch


# 定义自定义的数据集类来加载图像数据
class imgDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(index) + '.jpg')
        image = Image.open(img_path)
        label = np.array(self.annotations.iloc[index, 1])

        if self.transform:
            image = self.transform(image)

        return (image, label)
