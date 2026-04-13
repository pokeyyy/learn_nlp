# Word2Vec (Skip-gram with Softmax)

## 概述

Word2Vec是一种用于学习词向量（词嵌入）的浅层神经网络模型。它能够将高维的单词表示映射到低维的连续向量空间，保留词与词之间的语义和语法关系。

Word2Vec有两种主要架构：
- **Skip-gram**：给定一个词，预测其上下文词
- **CBOW (Continuous Bag of Words)**：给定上下文词，预测目标词

本实现采用的是**Skip-gram with Softmax**模型。

---

## 核心概念

### Skip-gram模型

Skip-gram模型的目标是：给定一个本地词（target word），预测它周围的上下文词（context words）。

**基本流程：**
1. 对于句子中的每个词，将其作为输入（目标词）
2. 预测该词左右固定窗口大小范围内的词（上下文词）
3. 通过最小化预测误差来学习词向量

**示例：**
```
句子: "apple banana fruit banana orange fruit"
窗口大小: 1 (左右各取1个词)

Skip-grams:
- (apple, banana)      → apple→banana
- (banana, apple)      → banana→apple
- (banana, fruit)      → banana→fruit
- (fruit, banana)      → fruit→banana
- (fruit, orange)      → fruit→orange
- (banana, fruit)      → banana→fruit
- (banana, orange)     → banana→orange
- (orange, banana)     → orange→banana
- (orange, fruit)      → orange→fruit
```

---

## 网络结构详解

### 整体架构

```
输入层 → 隐层 → 输出层
One-hot编码    词向量      Softmax概率分布
[voc_size]  [embedding_size]  [voc_size]
```

### 详细说明

#### 1. **输入层 (Input Layer)**
- 输入：One-hot编码的词向量
- 维度：`[batch_size, voc_size]`
- 说明：每个词用一个one-hot向量表示，只有对应位置为1，其余为0

#### 2. **隐层 (Hidden Layer / Embedding Layer)**
- 权重矩阵：`W` 
- 维度：`[voc_size, embedding_size]`
- 功能：将one-hot向量映射到低维连续空间
- 计算：`hidden = input × W`
- 输出维度：`[batch_size, embedding_size]`
- **关键点**：隐层的权重矩阵就是我们要学习的词向量矩阵

#### 3. **输出层 (Output Layer)**
- 权重矩阵：`WT`
- 维度：`[embedding_size, voc_size]`
- 功能：将词向量映射回词汇表大小的空间
- 计算：`output = hidden × WT`
- 输出维度：`[batch_size, voc_size]`
- 激活函数：通过CrossEntropyLoss内置的Softmax

#### 4. **特殊说明**
- `W` 和 `WT` **不是转置关系**，而是两个独立的权重矩阵
- `W`被用来学习"input"端的词向量（target word embedding）
- `WT`被用来学习"output"端的词向量（context word embedding）

### 矩阵维度变化流程

```
输入 One-hot       W矩阵         隐层向量        WT矩阵        输出Logits
[batch, voc]  × [voc, emb]  = [batch, emb]  × [emb, voc]  = [batch, voc]
  2 × 10       × 10 × 2      =  2 × 2       × 2 × 10      =  2 × 10
```

---

## 损失函数

### CrossEntropy Loss with Softmax

```
输出层 Logits
    ↓
  Softmax (将logits转换为概率分布)
    ↓
CrossEntropy Loss (计算预测分布与真实标签的交叉熵)
    ↓
反向传播更新权重
```

- **Softmax**：将输出logits转换为概率分布，使所有输出的和为1
- **CrossEntropyLoss**：衡量预测概率分布与真实标签的差异

---

## 训练过程

### 1. **数据准备**
- 输入句子列表
- 生成skip-gram对（目标词，上下文词）

### 2. **Batch采样**
- 从skip-grams中随机采样batch_size个样本
- 将目标词转换为one-hot编码
- 将上下文词标记保存为标签

### 3. **前向传播**
```
One-hot输入 → W矩阵 → 隐层词向量 → WT矩阵 → 输出Logits
```

### 4. **损失计算**
```
loss = CrossEntropyLoss(output, target_batch)
```

### 5. **反向传播与优化**
```
loss.backward()      # 计算梯度
optimizer.step()     # 更新参数（W和WT）
```

### 6. **重复迭代**
通过多个epoch的训练，不断优化词向量表示。

---

## 核心参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `batch_size` | 2 | 每次训练的样本数 |
| `embedding_size` | 2 | 词向量维度 |
| `epochs` | 5000 | 训练轮数 |
| `learning_rate` | 0.001 | Adam优化器的学习率 |
| `window_size` | 1 | Skip-gram窗口大小 |

---

## 输出与可视化

训练完成后，可以将学习到的词向量在低维空间（如2D）中可视化：

```python
W, WT = model.parameters()
for i, label in enumerate(word_list):
    x, y = W[0][i].item(), W[1][i].item()
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y))
```

结果展示：
- 语义相似的词会聚集在空间中接近的位置
- 例如"apple"、"banana"、"orange"会靠近在一起（都是水果）
- "dog"、"cat"、"monkey"也会靠近在一起（都是动物）

---

## 代码实现要点

### 1. random_batch函数
- 随机从skip_grams中采样batch_size个训练样本
- 返回one-hot编码的输入和对应的标签

### 2. Word2Vec类
- 继承`nn.Module`
- 包含两个线性层：`W`和`WT`
- forward函数定义前向传播过程

### 3. 主程序流程
1. 定义训练超参数
2. 准备训练数据和skip-grams
3. 初始化模型、损失函数和优化器
4. 训练循环：采样 → 前向传播 → 计算损失 → 反向传播 → 更新参数
5. 可视化结果

---

## 应用与扩展

### 应用场景
- **词相似度计算**：通过计算词向量之间的余弦相似度
- **词类比**：如"king - man + woman ≈ queen"
- **文本分类**：使用词向量作为特征
- **推荐系统**：计算物品之间的相似度

### 改进方向
- **Negative Sampling**：不计算所有词的概率，只计算正样本和负样本，加快训练
- **Hierarchical Softmax**：使用二叉树结构加快输出层计算
- **增大窗口**：考虑更广的上下文
- **子词信息**：使用FastText考虑字符n-gram信息

---

## 参考资源

- 原论文：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- 作者：Tomas Mikolov et al.
