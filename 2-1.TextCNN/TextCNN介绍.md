# TextCNN 介绍

本文基于 `2-1.TextCNN/TextCNN.py` ，详细说明经典 TextCNN 网络结构，并包含中文注释的关键代码说明。

## 1. TextCNN 概述

TextCNN（Yoon Kim, 2014）是用于文本分类的卷积神经网络。核心思想是使用不同大小的 1D 卷积核提取 n-gram 特征，接着通过最大池化聚合最显著特征，再拼接后送入全连接层进行分类。

本实现流程：
- Embedding 词向量
- 多尺寸卷积（卷积 + ReLU）
- Max-over-time 池化
- 拼接特征向量
- 全连接输出 logits

## 2. 代码结构及关键参数

- `embedding_size`: 词向量维度
- `sequence_length`: 输入文本固定序列长度
- `filter_sizes`: 卷积核尺寸列表，例如 `[2, 3, 4]` 表示 2-gram、3-gram、4-gram
- `num_filters`: 每种尺寸卷积核的数量
- `num_classes`: 分类类别数

## 3. 模型定义（`TextCNN` 类）

### 3.1 构造函数 `__init__`
- 词嵌入层：`nn.Embedding(vocab_size, embedding_size)`
- 全连接层：`nn.Linear(self.num_filters_total, num_classes, bias=False)`
- 偏置：`nn.Parameter(torch.ones([num_classes]))`
- 卷积层列表：`nn.ModuleList([...])`，每个 `nn.Conv2d(1, num_filters, (size, embedding_size))`

```python
self.num_filters_total = num_filters * len(filter_sizes)
self.W = nn.Embedding(vocab_size, embedding_size)
self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
self.Bias = nn.Parameter(torch.ones([num_classes]))
self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])
```

### 3.2 前向方法 `forward`
1. Embedding：`embedded_chars = self.W(X)`，输出形状 `[batch_size, sequence_length, embedding_size]`  
   *尺寸变化：输入 `[batch_size, sequence_length]` → `[6, 3, 2]`（基于样例参数）*

2. 增加通道维度：`embedded_chars.unsqueeze(1)` → `[batch, 1, sequence_length, embedding_size]`  
   *尺寸变化：`[6, 3, 2]` → `[6, 1, 3, 2]`（为 Conv2d 添加通道维度）*

3. 对每个卷积核尺寸（以 `filter_size=2` 为例）：  
   - 卷积 + ReLU: `h = F.relu(conv(embedded_chars))`  
     *尺寸变化：`[6, 1, 3, 2]` → `[6, 3, 2, 1]`（Conv2d(1, 3, (2, 2))，输出通道数为 num_filters=3）*  
   - MaxPool2d: `mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))`  
     *尺寸变化：`[6, 3, 2, 1]` → `[6, 3, 1, 1]`（池化窗口 (2, 1)，沿时间维度 max-over-time）*  
   - 池化并转置: `pooled = mp(h).permute(0, 3, 2, 1)`  
     *尺寸变化：`[6, 3, 1, 1]` → `[6, 1, 1, 3]`（转置为 [batch, num_filters, 1, 1] 格式）*

4. 拼接所有 `pooled_outputs`：`h_pool = torch.cat(pooled_outputs, len(filter_sizes))`  
   *尺寸变化：每个 pooled `[6, 1, 1, 3]`，拼接后 `[6, 1, 1, 9]`（9 = 3 * len(filter_sizes)）*

5. 拉平：`h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])`  
   *尺寸变化：`[6, 1, 1, 9]` → `[6, 9]`（展平为全连接层输入）*

6. 线性分类 + 偏置：`model = self.Weight(h_pool_flat) + self.Bias`  
   *尺寸变化：`[6, 9]` → `[6, 2]`（输出 logits，对应 num_classes=2）*


## 4. 样例流程（`__main__`）

### 4.1 词表构建
- 将样本 `sentences` 展开成 `word_list`
- 生成 `word_dict`，并得到 `vocab_size`

### 4.2 张量准备
- `inputs`：用 `word_dict` 转为 `LongTensor`
- `targets`：标签 `LongTensor`

### 4.3 损失与优化
- 损失：`nn.CrossEntropyLoss()`（接收 logits 和整型标签）
- 优化：`optim.Adam(model.parameters(), lr=0.001)`

### 4.4 训练循环
- `optimizer.zero_grad()`
- 前向计算 `output = model(inputs)`
- 计算 `loss` 并 `loss.backward()`
- `optimizer.step()`

### 4.5 测试与预测
- 输入 `test_text` 分词转索引
- `predict = model(test_batch).data.max(1, keepdim=True)[1]`
- 类别分别映射为 “Good” 或 “Bad”

## 5. TextCNN 网络结构图

- 输入：`[batch, sequence_length]`
- Embedding：`[batch, sequence_length, embedding_size]`
- 增加 channel：`[batch, 1, sequence_length, embedding_size]`
- 卷积（多核）与 ReLU
- Max-over-time 池化，得到 `num_filters` 特征向量
- 拼接后 `num_filters_total` 全连接，与 `bias` 求 logits

## 6. 中文注释说明

代码中已将所有英文注释替换为中文，确保每行逻辑清晰可读，便于学习和修改。以下是部分已更新内容：
- `# 增加通道维度 1`（Embedding -> Conv2d 输入）
- `# 最大池化操作`（不同 filter_size 的 MaxPool2d）
- `# 交叉熵损失函数期望整数标签（非 one-hot）`

## 7. 可扩展建议

1. 支持可变长度输入（通过 padding 或变长批处理）
2. 增加 Dropout 防止过拟合
3. 试验不同 `filter_sizes`（2,3,4,5）与 `num_filters`
4. 使用预训练词向量如 GloVe、word2vec
5. 添加 k-max pooling 或 attention 提升效果
