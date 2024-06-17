# ͼ��ʶ��������
---

## ��Ŀ����

����������ҵ��Ŀ���ͼ��ʶ����������硣
��Fashion-MNIST���ݼ��Ͻ���ѵ����

---

## ��������

���軷�����£��������л��ն������У�

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

## ��Ŀ�ṹ

* **/data**����ѵ���Ͳ��Ե����ݣ���**���ݴ���**���֣���
* **/models**����ѵ�����ģ�ͣ�
* **/result_img**��ѵ���Ͳ��Խ���Ŀ��ӻ���
* **train.py**��ѵ���ű���**test.py**�ǲ��Խű���
* **img_dataset.py**�����ݼ��࣬**img_NN.py**�������磻
* **train_data_process.py**��**test_data_process.py**�����ݼ�����ű���

---

## ���ݴ���

���õ������ݼ�Fashion-MNIST�����ͼƬ�������ݼ�����10������ʱװͼ��ѵ������ 60,000 ��ͼƬ�����Լ�����10,000��ͼƬ��
ͼƬΪ�Ҷ�ͼƬ���߶ȺͿ�Ⱦ�Ϊ28���أ�ͨ������channel��Ϊ1��10�����ֱ�Ϊ��

* t-shirt��T����
* trousers�����ӣ�
* pullover��������
* dress������ȹ��
* coat�����ף�
* sandal����Ь��
* shirt��������
* sneaker���˶�Ь��
* bag������
* ankle boot����ѥ��

ʹ��ѵ�������ݽ���ѵ�������Լ����ݽ��в��ԡ�
���ݼ������������£�
[GitHub - zalandoresearch/fashion-mnist: A MNIST-like fashion product database. Benchmark](https://github.com/zalandoresearch/fashion-mnist)

���ص������ݼ�Ϊ�ĸ�ѹ���ļ�:

* train-images-idx3-ubyte.gz
* train-labels-idx1-ubyte.gz
* t10k-images-idx3-ubyte.gz
* t10k-labels-idx1-ubyte.gz

������ڸ�Ŀ¼�µ�/org_data�У�����������ݴ���ű���

```bash
python train_dat_process.py
python test_dat_process.py
```

---

## ѵ��

����img_NN.py�д���ҵ�ͼ��ʶ�������硣

���Ե��ŻҶ�ͼ��Ϊ���룬������������������ȫ���Ӳ㡣���У�

* **�����_1��** ����ͨ��Ϊ1�����ͨ��Ϊ32������˴�СΪ3��
* **�����_2��** ����ͨ��Ϊ32�����ͨ��Ϊ64������˴�СΪ3x3��
* **ȫ���Ӳ�_1��** �����СΪ64*7*7�������СΪ128
* **ȫ���Ӳ�_2��** �����СΪ128�������СΪ10����Ӧ10���������

��train.py��ѵ��ͼ���������硣

```bash
python train.py
```

ѵ�������У���ʧ����ѡ�񽻲�����ʧ���Ż���ѡ�þ����Adam�Ż�����ѧϰ����Ϊ0.001�����δ�СΪ64��

ѵ�������󣬽���ʧ���߿��ӻ���ͼ��洢��/result_img��ģ������ΪImageClassifier_����ǰʱ�����ڣ�.ckpt��

---

## ����

��test.py�н��в��ԡ�

```bash
python test.py
```

����ģ�Ͳ������ò��Լ����ݽ��в��ԡ�

������ʹ��Ԥ�������ƻ������󣬽��п��ӻ�չʾ�����档

---

## �������

ѵ�����̵���ʧ�������ߺͻ��������λ�� **/result_img** ��

ѵ���������ģ�ͱ����� **/models** �ڡ�

