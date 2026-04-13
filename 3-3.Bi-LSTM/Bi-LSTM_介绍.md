# Bi-LSTM (双向长短期记忆网络)

## 1. 概述

Bi-LSTM（Bidirectional LSTM, 双向长短期记忆网络）是在标准LSTM的基础上进行的改进，通过同时从前向和后向两个方向处理序列数据，使模型能够捕捉更完整的上下文信息。

### 与LSTM的主要区别
- **单向LSTM**：只能获取过去时刻的信息（前向依赖）
- **Bi-LSTM**：同时获取过去和未来的信息（双向依赖）

## 2. 网络结构

### 2.1 总体架构

```
前向LSTM ──┐
           ├─→ 拼接 ──→ 线性层 ──→ 输出层
后向LSTM ──┘
```

### 2.2 详细的网络参数说明

#### 2.2.1 输入层
- **输入维度**：`(batch_size, sequence_length, n_class)`
  - `batch_size`：批次大小（处理的样本个数）
  - `sequence_length`：序列长度（时间步数）
  - `n_class`：特征维度（每个词的One-hot编码维度）

在实现中，PyTorch的LSTM期望输入形状为 `(sequence_length, batch_size, input_size)`，因此需要进行转置。

#### 2.2.2 LSTM层（双向）

```python
self.lstm = nn.LSTM(
    input_size=n_class,           # 输入特征维度
    hidden_size=n_hidden,         # 隐藏层维度
    bidirectional=True,           # 启用双向
    num_layers=1                  # LSTM层数
)
```

**关键参数说明**：
- `input_size`：输入特征维度
- `hidden_size`：每个方向的隐藏单元数（`n_hidden`）
- `bidirectional=True`：启用双向模式
  - 前向LSTM：从左到右处理序列
  - 后向LSTM：从右到左处理序列

**状态初始化**：
```python
hidden_state = torch.zeros(1*2, len(X), n_hidden)   # [2, batch_size, n_hidden]
cell_state = torch.zeros(1*2, len(X), n_hidden)     # [2, batch_size, n_hidden]
```

其中：
- `1 * 2`：表示1层 × 2个方向（前向+后向）
- 第一维：`2`代表前向和后向两个LSTM
- 第二维：`batch_size`
- 第三维：`n_hidden`隐藏单元数

**输出维度**：
- 形状：`(sequence_length, batch_size, n_hidden * 2)`
- 最后一步输出：`(batch_size, n_hidden * 2)`
- 拼接的原因：前向LSTM输出（n_hidden）+ 后向LSTM输出（n_hidden）= n_hidden * 2

#### 2.2.3 全连接层（输出层）

```python
self.W = nn.Linear(n_hidden * 2, n_class, bias=False)
self.b = nn.Parameter(torch.ones([n_class]))
```

**参数说明**：
- 输入维度：`n_hidden * 2`（来自双向LSTM拼接的输出）
- 输出维度：`n_class`（词表大小）
- 最终输出：`(batch_size, n_class)`

## 3. 前向传播过程

```python
def forward(self, X):
    # 步骤1：转置输入维度
    input = X.transpose(0, 1)  # [n_step, batch_size, n_class]
    
    # 步骤2：初始化隐状态和单元状态
    hidden_state = torch.zeros(1*2, len(X), n_hidden)
    cell_state = torch.zeros(1*2, len(X), n_hidden)
    
    # 步骤3：通过双向LSTM
    outputs, (final_hidden, final_cell) = self.lstm(input, (hidden_state, cell_state))
    # outputs: [n_step, batch_size, n_hidden * 2]
    
    # 步骤4：提取最后一个时间步的输出
    outputs = outputs[-1]  # [batch_size, n_hidden * 2]
    
    # 步骤5：通过全连接层
    model = self.W(outputs) + self.b  # [batch_size, n_class]
    
    return model
```

**关键步骤**：
1. **转置**：将输入从 `(batch, seq_len, features)` 转为 `(seq_len, batch, features)`
2. **并行处理**：双向LSTM同时从前后两个方向处理序列
3. **特征融合**：将前向和后向的隐状态拼接
4. **序列聚合**：只取最后一个时间步的输出作为全局特征
5. **预测**：通过线性变换得到每个类别的得分

## 4. 数据流示例

假设参数设置：
- `n_hidden = 5`（隐藏单元数）
- `n_class = 10`（词表大小）
- `batch_size = 32`
- `max_len = 20`（最大序列长度）

**数据流向**：

```
输入: [32, 20, 10]
    ↓
转置: [20, 32, 10]
    ↓
双向LSTM处理:
  ├─ 前向LSTM输出: [20×32×5]
  └─ 后向LSTM输出: [20×32×5]
    ↓
拼接: [20, 32, 10]
    ↓
取最后一步: [32, 10]
    ↓
线性层: [32, 10]  (W.shape = [10, 10], b.shape = [10])
    ↓
输出: [32, 10]（每个样本10个类别的预测分数）
```

## 5. 应用场景

Bi-LSTM在NLP任务中的典型应用：

| 应用场景 | 说明 |
|---------|------|
| **文本分类** | 利用完整的上下文信息对文本进行分类 |
| **序列标注** | NER（命名实体识别）、词性标注等 |
| **语义相似度** | 获取整句的全局语义表示 |
| **下一词预测** | 本实现的示例应用 |
| **机器翻译** | 作为编码器获取完整的源语言表达 |

## 6. 本实现中的具体应用：下一词预测

### 任务描述
给定前n个词，预测第n+1个词

### 数据处理
```python
sentence = "Lorem ipsum dolor sit amet..."
# 为每个词创建一个训练样本
# 输入：前1个词 → 预测第2个词
# 输入：前2个词 → 预测第3个词
# ...
```

### 模型工作流程
1. **编码**：每个词转为One-hot编码
2. **序列处理**：Bi-LSTM学习上下文模式
3. **预测**：输出下一个最可能出现的词

## 7. 与TextRNN的对比

| 特性 | TextRNN | Bi-LSTM |
|------|---------|---------|
| 处理方向 | 单向（前向） | 双向 |
| 上下文 | 仅过去信息 | 完整上下文 |
| 表达能力 | 较弱 | 较强 |
| 计算复杂度 | 低 | 高 |
| 适用任务 | 实时预测 | 离线分析 |

## 8. 优缺点总结

### 优点
✓ 能够捕捉完整的上下文信息  
✓ 对长距离依赖有更强的学习能力  
✓ 在许多NLP任务上性能优于单向LSTM  
✓ 双向信息融合使表达更加丰富  

### 缺点
✗ 计算复杂度更高（双向处理）  
✗ 不适合实时预测场景（需要未来信息）  
✗ 内存占用增加  
✗ 训练时间更长  

## 9. 关键代码解析

### 关键参数含义
```python
# 隐状态维度计算
hidden_state.shape = [num_layers * num_directions, batch_size, hidden_size]
                   = [1 * 2, 32, 5]  = [2, 32, 5]

# LSTM输出维度
lstm_output.shape = [seq_len, batch_size, hidden_size * num_directions]
                  = [20, 32, 5 * 2] = [20, 32, 10]

# 取最后时戳
last_output.shape = [batch_size, hidden_size * num_directions]
                  = [32, 10]

# 线性变换
final_output.shape = [batch_size, num_classes]
                   = [32, 10]
```

## 10. 总结

Bi-LSTM是标准LSTM的重要变体，通过双向处理序列，使模型能够获取更完整的上下文信息。虽然计算成本更高，但在需要全局语义理解的任务中，Bi-LSTM的性能通常优于单向LSTM，是现代NLP模型中的重要组件。
