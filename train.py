# -* coding=utf8 *-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from img_NN import ImageClassifier
from img_dataset import imgDataset
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# 定义图像转换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图像
    transforms.ToTensor(),
])

# 创建数据集
train_dataset = imgDataset(csv_file='data/train_labels.csv', img_dir='data/train_img', transform=transform)

# 定义超参数
learning_rate = 0.001
num_epochs = 5
batch_size = 64

# 创建数据加载器
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

# 初始化神经网络模型
# model = ImageClassifier(num_qubits, num_layers)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageClassifier().to(device)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 存储每一步的损失值
step_losses = []

# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')):
        # 将数据移至 GPU
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        step_losses.append(loss.item())  # 记录每一步的损失值


        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # 计算平均损失值
    avg_epoch_loss = epoch_loss / len(train_loader)
    # step_losses.append(avg_epoch_loss)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}')



# 保存模型
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_path = f'models/ImageClassifier_{current_time}.ckpt'# 绘制验证时的损失曲线
plt.plot(range(1, len(step_losses) + 1), step_losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Validation Loss Curve')

plt.savefig(f'result_img/validation_loss_curve_{current_time}.png')

plt.show()

os.makedirs('result_img', exist_ok=True)
os.makedirs('models', exist_ok=True)


torch.save(model.state_dict(), model_path)
