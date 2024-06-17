# -* coding=utf8 *-
import gzip
import numpy as np
import os
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt



# 读取图像数据
with gzip.open('org_data/train-images-idx3-ubyte.gz') as all_img:
    all_img = all_img.read()

# 读取标签数据
with gzip.open('org_data/train-labels-idx1-ubyte.gz') as all_lbl:
    all_lbl = all_lbl.read()

# 提取图像数据
image_size = 28 * 28
num_images = 60000
all_img = np.frombuffer(all_img, dtype=np.uint8, offset=16)  # 将字节数据转换为NumPy数组
all_img = all_img.reshape(num_images, image_size)  # 将数据reshape为图像的形状

# img = all_img[0].reshape(28, 28)
# plt.imshow(img)
# plt.show()
#
# # breakpoint()


# 创建保存图像的文件夹
os.makedirs('data/train_img', exist_ok=True)

for i in range(num_images):
    img = all_img[i].reshape(28, 28)
    img = Image.fromarray(img, mode='L')
    img.save(f'data/train_img/{i}.jpg')
# breakpoint()


# 将标签数据转换为合适的数据类型
all_lbl = np.frombuffer(all_lbl, dtype=np.uint8)

# 处理标签数据并保存为CSV表格
labels = [int(label) for label in all_lbl[8:]]

print("labels[0] = ", labels[0])
df = pd.DataFrame({'ImageId': range(1, num_images + 1), 'Label': labels})
df.to_csv('data/train_labels.csv', index=False)
