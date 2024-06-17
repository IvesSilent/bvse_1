# 图像识别神经网络
---

## 项目简述

本人用于作业项目搭建的图像识别分类神经网络。
在Fashion-MNIST数据集上进行训练。

---

## 环境配置

所需环境如下，在命令行或终端中运行：

```bash
pip install torch==2.1.2+cu118
pip install Pillow==9.4.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install torchvision==0.16.2+cu118
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install scikit-learn==1.3.0
pip install tqdm==4.65.0
```

---

## 项目结构

* **/data**中是训练和测试的数据（见**数据处理**部分）；
* **/models**中是训练后的模型；
* **/result_img**是训练和测试结果的可视化；
* **train.py**是训练脚本，**test.py**是测试脚本；
* **img_dataset.py**是数据集类，**img_NN.py**是神经网络；
* **train_data_process.py**和**test_data_process.py**是数据集处理脚本。

---

## 数据处理

采用的是数据集Fashion-MNIST。这个图片分类数据集包含10个类别的时装图像。训练集有 60,000 张图片，测试集中有10,000张图片。
图片为灰度图片，高度和宽度均为28像素，通道数（channel）为1。10个类别分别为：

* t-shirt（T恤）
* trousers（裤子）
* pullover（套衫）
* dress（连衣裙）
* coat（外套）
* sandal（凉鞋）
* shirt（衬衫）
* sneaker（运动鞋）
* bag（包）
* ankle boot（短靴）

使用训练集数据进行训练，测试集数据进行测试。
数据集下载链接如下：
[GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark](https://github.com/zalandoresearch/fashion-mnist)

下载到的数据集为四个压缩文件:

* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz

将其放在根目录下的/org_data中，随后运行数据处理脚本。

```bash
python train_dat_process.py
python test_dat_process.py
```

---

## 训练

我在img_NN.py中搭建了我的图像识别神经网络。

它以单张灰度图像为输入，共有两个卷积层和两个全连接层。其中：

* **卷积层_1：** 输入通道为1，输出通道为32，卷积核大小为3；
* **卷积层_2：** 输入通道为32，输出通道为64，卷积核大小为3x3；
* **全连接层_1：** 输入大小为64*7*7，输出大小为128
* **全连接层_2：** 输入大小为128，输出大小为10（对应10个数字类别）

在train.py中训练图分类神经网络。

```bash
python train.py
```

训练过程中，损失函数选择交叉熵损失，优化器选用经典的Adam优化器。学习率设为0.001，批次大小为64。

训练结束后，将损失曲线可视化，图像存储于/result_img；模型命名为ImageClassifier_（当前时间日期）.ckpt。

---

## 测试

在test.py中进行测试。

```bash
python test.py
```

加载模型参数后，用测试集数据进行测试。

结束后，使用预测结果绘制混淆矩阵，进行可视化展示并保存。

---

## 结果保存

训练过程的损失函数曲线和混淆矩阵均位于 **/result_img** 。

训练结束后的模型保存于 **/models** 内。

