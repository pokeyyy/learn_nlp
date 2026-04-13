# NNLM (Neural Network Language Model) 详解

## 1. 简介

NNLM（神经网络语言模型）是由Bengio等人在2003年提出的经典模型，开启了使用神经网络进行自然语言处理的新时代。相比传统的n-gram语言模型，NNLM通过学习词向量表示（word embeddings），能够更好地捕捉词之间的语义关系。

### 核心思想
- **词向量表示**：将离散的词汇映射到连续的低维向量空间（embedding space）
- **上下文预测**：根据前n-1个词预测第n个词
- **参数共享**：所有词共享同一套embedding参数

## 2. 模型架构

### 2.1 网络结构图

```
输入词序列
    |
    v
┌─────────────────────┐
│  Embedding层 (C)    │  将词汇转换为m维向量
│  维度：V × m        │  V = 词汇表大小, m = embedding大小
└─────────────────────┘
    |
    v
┌─────────────────────┐
│   Flatten/Concat    │  展平为(n_step × m)维向量
│  维度：n_step × m   │
└─────────────────────┘
    |
    +─────────────────────+
    |                     |
    v                     v
┌──────────────────┐  ┌──────────────────┐
│  隐藏层 (H + d)  │  │  直连层 (W)      │
│  H: Linear层     │  │  W: Linear层     │
│  激活: tanh      │  │  无激活函数      │
│  输出：n_hidden  │  │  输出：n_class   │
└──────────────────┘  └──────────────────┘
    |                     |
    v                     v
┌──────────────────────────────────────┐
│  U层 + 加法                          │
│  U: 将tanh输出映射到n_class          │
│  + b (偏置)                          │
└──────────────────────────────────────┘
    |
    v
┌──────────────────────────────────────┐
│  输出层：词汇概率分布                 │
│  维度：n_class                       │
└──────────────────────────────────────┘
```

### 2.2 详细层级说明

#### 输入层（Input Layer）
- **输入**：文本序列的前n-1个单词的索引
- **维度**：[batch_size, n_step]
- **例子**：对于句子"i like dog"，输入为["i", "like"]

#### C层 - 词向量映射（Embedding Layer）
```python
self.C = nn.Embedding(n_class, m)
```
- **作用**：将词汇索引转换为m维的密集向量
- **参数**：
  - `n_class`：词汇表大小（V）
  - `m`：embedding维度（论文中标记为m）
- **输出维度**：[batch_size, n_step, m]
- **数学表示**：$C(w)$表示词w的embedding向量

#### H层 - 隐藏层（Hidden Layer）
```python
self.H = nn.Linear(n_step * m, n_hidden, bias=False)
self.d = nn.Parameter(torch.ones(n_hidden))
```
- **作用**：处理拼接后的embedding向量，提取高层特征
- **参数**：
  - `H`：权重矩阵，形状为(n_step × m, n_hidden)
  - `d`：偏置向量，形状为(n_hidden,)
- **激活函数**：tanh
- **输出维度**：[batch_size, n_hidden]
- **数学表示**：$h = tanh(d + H \cdot x)$

其中 $x$ 是拼接后的embedding向量

#### W层 - 直连层（Short-cut Connection）
```python
self.W = nn.Linear(n_step * m, n_class, bias=False)
```
- **作用**：允许输入直接连接到输出，相当于skip connection
- **参数**：权重矩阵，形状为(n_step × m, n_class)
- **输出维度**：[batch_size, n_class]
- **数学表示**：$Wx$

#### U层 - 输出映射（Output Layer - Hidden Part）
```python
self.U = nn.Linear(n_hidden, n_class, bias=False)
```
- **作用**：将隐藏层输出映射到词汇表大小
- **参数**：权重矩阵，形状为(n_hidden, n_class)
- **输出维度**：[batch_size, n_class]
- **数学表示**：$U \cdot h$

#### 输出层
```python
output = self.b + self.W(X) + self.U(tanh)
```
- **作用**：计算最终的分数（logits）
- **偏置**：b，形状为(n_class,)
- **完整公式**：$y = b + Wx + U \cdot tanh(d + Hx)$

其中：
- $x$ 是输入embedding的拼接向量
- $W$ 提供直接的快速连接
- $U \cdot tanh(d + Hx)$ 提供非线性的高级特征提取

## 3. 前向传播过程

```
输入 X: [batch_size, n_step]  # 词索引序列
   ↓
X = C(X)  # embedding: [batch_size, n_step, m]
   ↓
X = X.flatten()  # 展平: [batch_size, n_step*m]
   ↓
隐藏激活: h = tanh(d + H(X))  # [batch_size, n_hidden]
   ↓
输出: Y = b + W(X) + U(h)  # [batch_size, n_class]
   ↓
结果：词汇表上的概率分布（未归一化的logits）
```

## 4. 关键参数设置

| 参数名 | 符号 | 说明 | 默认值 |
|--------|------|------|--------|
| n_step | n-1 | 使用的前向词的个数 | 2 |
| m | m | embedding向量维度 | 2 |
| n_hidden | h | 隐藏层维度 | 2 |
| n_class | V | 词汇表大小 | 根据数据集 |

### 参数数量统计

总参数数 = 
- Embedding: $V \times m$
- H层: $(n_{step} \times m) \times n_{hidden}$
- d: $n_{hidden}$
- U层: $n_{hidden} \times V$
- W层: $(n_{step} \times m) \times V$
- b: $V$

**总计**：$Vm + nm \cdot h + h + hV + nmV + V$

其中n = n_step

## 5. 训练过程

### 5.1 损失函数
使用**交叉熵损失**（Cross Entropy Loss）：

$$L = -\sum_{i=1}^{batch\_size} \log P(y_i | x_i)$$

其中 $P(y_i | x_i) = \frac{e^{y_i}}{\sum_j e^{y_j}}$ (softmax)

### 5.2 优化算法
- 使用Adam优化器
- 学习率：0.001
- 训练轮数：5000 epochs

### 5.3 预测
训练完成后，对于输入的前n-1个词，模型选择输出概率最高的词作为预测结果：

$$\hat{w}_n = \arg\max_w P(w | w_1, ..., w_{n-1})$$

## 6. 代码示例详解

### 数据准备
```python
# 句子集合
sentences = ["i like dog", "i love coffee", "i hate milk"]

# 构建词汇表和映射
word_dict = {"i": 0, "like": 1, "dog": 2, ...}  # 词→索引
number_dict = {0: "i", 1: "like", 2: "dog", ...}  # 索引→词
```

### 批量处理
对于每个句子，提取：
- **输入**：前n-1个词的索引
- **目标**：第n个词的索引

例如："i like dog" 
- 输入：[词典["i"], 词典["like"]]
- 目标：词典["dog"]

### 训练循环
1. 前向传播：计算模型输出
2. 计算损失：交叉熵
3. 反向传播：计算梯度
4. 优化步骤：更新权重

## 7. 优缺点分析

### 优点
✓ **简单高效**：结构清晰，易于实现和理解
✓ **参数共享**：词向量在整个模型中共享，减少参数量
✓ **可解释性强**：学到的embedding向量可以反映词义
✓ **基础性**：后续Word2Vec、BERT等模型都基于此思想

### 缺点
✗ **固定上下文长度**：只能使用前n-1个词
✗ **性能局限**：由于规模限制，性能不如现代模型
✗ **样本需求**：需要较大的训练集来学习好的embedding
✗ **OOV问题**：无法处理训练集外的词汇

## 8. 后续发展

NNLM为后续多个重要模型奠定了基础：

- **Word2Vec (2013)**：采用更高效的训练方式（Skip-gram, CBOW）
- **Sequence Models**：RNN、LSTM等处理可变长度序列
- **Attention Mechanism**：更灵活地捕捉长程依赖
- **BERT (2018)**：双向预训练，刷新多项NLP任务记录
- **GPT系列**：大规模预训练，生成式语言模型

## 9. 总结

NNLM是现代NLP的奠基之作，引入了以下关键概念：
1. **分布式表示**（Distributed Representation）：词用密集向量表示
2. **词向量（Word Embeddings）**：可学习的词语表示
3. **神经网络语言模型**：用深度学习处理语言建模任务

尽管现在有更先进的模型，但理解NNLM的原理对于学习现代NLP技术至关重要。
