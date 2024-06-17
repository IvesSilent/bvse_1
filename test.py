# -* coding=utf8 *-
import torch
from torchvision import transforms
from img_dataset import imgDataset
from img_NN import ImageClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime


# 定义图像转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# 初始化神经网络模型并加载训练好的参数
batch_size = 64
model = ImageClassifier()
model.load_state_dict(torch.load('models/ImageClassifier_2024-06-17_19-59-02.ckpt'))  # 加载训练好的模型参数

# 创建测试数据集
test_dataset = imgDataset(csv_file='data/test_labels.csv', img_dir='data/test_img', transform=transform)

# 创建数据加载器
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 测试模型
model.eval()  # 将模型设置为评估模式
predictions = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 可视化预测结果
class_names = [str(i) for i in range(10)]  # 假设标签从0到9
conf_matrix = confusion_matrix(true_labels, predictions)
# conf_matrix = conf_matrix / conf_matrix.astype(np.float).sum(axis=1)  # 转换为百分比
conf_matrix = conf_matrix / conf_matrix.astype(float).sum(axis=1)  # 转换为百分比

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt=".2%", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f'result_img/confusion_matrix_{current_time}.png')
plt.show()

