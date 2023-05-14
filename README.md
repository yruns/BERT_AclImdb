基于BERT, BiLSTM在AclImdb数据集上的情感分类实验
================
## 人工智能及应用课程作业 by 乐书豪 202031119020167


## 使用方法

1. 将`bert-base-uncase`下载到本地
2. 运行`python train_teacher.py`训练teacher模型，并保存至`checkpoints/`
3. 运行`python distil.py`进行蒸馏，并保存至`checkpoints/`

训练日志保存至`logs/` <br>
可以通过调整`utils/hyperParams.py`中的参数来调整蒸馏模型的参数<br>
`tempurature`和`alpha`参数会显著影响蒸馏效果<br>

