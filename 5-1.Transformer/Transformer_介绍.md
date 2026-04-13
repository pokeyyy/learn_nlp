# Transformer 模型详解

## 概述

Transformer 是由 Vaswani 等人在 2017 年论文《Attention Is All You Need》中提出的一种序列到序列（Seq2Seq）模型。与传统的 RNN/LSTM 不同，Transformer 完全基于**自注意力机制（Self-Attention）**，摒弃了循环结构，实现了高度并行化的训练。

## 核心优势

- **并行计算**：不依赖时间步序列，可同时处理整个序列
- **长距离依赖**：自注意力机制可直接捕捉任意位置间的依赖关系
- **计算效率高**：相比 RNN 的 O(n) 序列操作，自注意力为 O(1) 序列操作

## 整体架构

Transformer 采用经典的**编码器-解码器（Encoder-Decoder）**结构：

```
输入序列 → [编码器] → 中间表示 → [解码器] → 输出序列
```

- **编码器（Encoder）**：由 N 个相同的层堆叠而成（论文中 N=6）
- **解码器（Decoder）**：同样由 N 个相同的层堆叠而成
- 每层包含两个核心子层：多头注意力机制和前馈神经网络

## 详细组件

### 1. 输入嵌入（Input Embedding）

将离散的词元（token）映射为连续的向量表示：

```
Embedding: vocab_size × d_model
```

示例中 `d_model = 512` 是词向量的维度。

### 2. 位置编码（Positional Encoding）

由于 Transformer 没有循环结构，无法天然感知位置信息，需要显式地加入位置编码。

本实现采用**正弦/余弦位置编码**：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- `pos`：词在序列中的位置
- `i`：维度索引

这种编码方式的优势是：模型可以学习 attend to 相对位置，因为对于固定的偏移量 k，PE(pos+k) 可以表示为 PE(pos) 的线性函数。

最终输入 = 词嵌入 + 位置编码

### 3. 注意力掩码（Attention Mask）

#### 3.1 填充掩码（Pad Mask）

用于忽略填充 token（PAD）的注意力。当批次中序列长度不一致时，较短的序列会用 PAD 填充，这些位置不应参与注意力计算。

```python
# 判断哪些是 PAD token（值为0），生成掩码
pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
```

#### 3.2 后续掩码（Subsequent Mask）

用于解码器的自注意力层，防止位置 i 注意到位置 i 之后的信息（防止信息泄露），实现自回归生成。

```python
# 生成上三角掩码
subsequent_mask = np.triu(np.ones(attn_shape), k=1)
```

### 4. 缩放点积注意力（Scaled Dot-Product Attention）

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

**为什么要缩放？** 当 d_k 较大时，点积结果会很大，导致 softmax 梯度接近于 0。除以 sqrt(d_k) 可以使梯度稳定。

```python
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)  # 掩码位置填充极小值
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
```

### 5. 多头注意力（Multi-Head Attention）

将 Q、K、V 分别通过不同的线性投影映射到 h 个子空间，每个子空间独立计算注意力，最后拼接所有头的输出。

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

本实现中 `n_heads = 8`，每个头的维度 `d_k = d_v = 64`，因此总维度为 8 × 64 = 512 = d_model。

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
```

多头注意力的关键特点：
- 允许多个不同的注意力子空间
- 每个头可以关注不同的语义信息（如语法、语义、长距离依赖等）
- 使用残差连接（residual connection）和层归一化（Layer Normalization）

### 6. 位置前馈网络（Position-wise Feed-Forward Network）

对序列中的每个位置独立应用相同的全连接网络：

```
FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2
```

本实现使用两个 1x1 卷积来实现：

```python
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
```

其中 `d_ff = 2048` 是中间层的维度，远大于 d_model，提供更大的表征能力。

### 7. 编码器层（Encoder Layer）

每个编码器层由两个子层组成：

```
输入 → [多头自注意力] → [残差连接 + 层归一化] → [前馈网络] → [残差连接 + 层归一化] → 输出
```

```python
class EncoderLayer(nn.Module):
    def __init__(self):
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn
```

编码器中的所有 Q、K、V 都来自同一输入序列（自注意力）。

### 8. 解码器层（Decoder Layer）

每个解码器层由**三个**子层组成：

```
输入 → [掩码多头自注意力] → [残差连接 + 层归一化]
     → [编码器-解码器注意力] → [残差连接 + 层归一化]
     → [前馈网络] → [残差连接 + 层归一化] → 输出
```

```python
class DecoderLayer(nn.Module):
    def __init__(self):
        self.dec_self_attn = MultiHeadAttention()      # 解码器自注意力
        self.dec_enc_attn = MultiHeadAttention()        # 编码器-解码器注意力
        self.pos_ffn = PoswiseFeedForwardNet()
    
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn
```

**两种注意力的区别**：
- **解码器自注意力**：Q 来自解码器，K、V 也来自解码器（关注已生成的输出）
- **编码器-解码器注意力**：Q 来自解码器，K、V 来自编码器（关注输入序列的相关信息）

### 9. 完整编码器（Encoder）

```python
class Encoder(nn.Module):
    def __init__(self):
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model), freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])  # n_layers = 6
```

### 10. 完整解码器（Decoder）

```python
class Decoder(nn.Module):
    def __init__(self):
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model), freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])  # n_layers = 6
```

### 11. 输出投影（Projection）

解码器的最终输出通过一个线性层投影到目标词表大小：

```python
self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
```

## 两种推理方式

### 1. Teacher Forcing（训练时）

训练时使用真实的输出序列作为解码器输入：

```
输入：S i want a beer
期望输出：i want a beer E
```

### 2. 贪婪解码（Greedy Decoder，推理时）

在推理时，逐步生成每个词，将上一步的输出作为下一步的输入：

```python
def greedy_decoder(model, enc_input, start_symbol):
    enc_outputs, _ = model.encoder(enc_input)
    dec_input = torch.zeros(1, 5).type_as(enc_input.data)
    next_symbol = start_symbol  # 从起始符号 S 开始
    for i in range(0, 5):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input
```

## 模型参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 512 | 词嵌入维度 |
| d_ff | 2048 | 前馈网络中间层维度 |
| d_k = d_v | 64 | 注意力头维度 |
| n_layers | 6 | 编码器和解码器的层数 |
| n_heads | 8 | 多头注意力的头数 |

## 残差连接与层归一化

每个子层的输出都采用**残差连接（Residual Connection）**和**层归一化（Layer Normalization）**：

```
Output = LayerNorm(x + Sublayer(x))
```

这种结构的优势：
- **残差连接**：有助于梯度的反向传播，缓解梯度消失问题
- **层归一化**：使每个样本的特征分布标准化，加速训练

## 训练过程

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    optimizer.zero_grad()
    outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    loss = criterion(outputs, target_batch.contiguous().view(-1))
    loss.backward()
    optimizer.step()
```

## 注意力可视化

训练完成后可通过可视化工具查看注意力权重，了解模型关注输入序列的哪些部分：

```python
def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax.matshow(attn, cmap='viridis')
    plt.show()
```

## 总结

Transformer 的核心创新在于：
1. **完全基于注意力**：摒弃了 RNN 的循环计算
2. **多头注意力**：多子空间捕获不同的依赖关系
3. **位置编码**：解决了序列位置信息问题
4. **残差连接**：保证了信息的有效流通，使深度网络的训练成为可能

这些设计使得 Transformer 在机器翻译、文本生成等 NLP 任务上取得了当时的最佳效果（SOTA），并成为后续 BERT、GPT 等预训练模型的基础架构。
