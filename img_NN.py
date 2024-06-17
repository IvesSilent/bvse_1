# -* coding=utf8 *-
import torch.nn as nn
import torch



class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # 输入通道为1（灰度图像），输出通道为32，卷积核大小为3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 输入通道为32，输出通道为64，卷积核大小为3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层，输入大小为64*7*7，输出大小为128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 全连接层，输入大小为128，输出大小为10（对应10个数字类别）
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)  # 将卷积层的输出展平成一维向量
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x
