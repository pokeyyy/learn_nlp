import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# S: 解码器输入的起始符号 (Start)
# E: 解码器输出的结束符号 (End)
# P: 当批次数据长度不足时用于填充序列的占位符 (Pad)，其词表索引必须为 0


def make_batch(sentences):
    """
    将原始句子文本转换为模型所需的批次张量
    :param sentences: 包含 [源序列, 目标输入(含S), 目标标签(含E)] 的列表
    :return: (enc_inputs, dec_inputs, target_batch) 形状均为 (batch_size, seq_len)
    """
    # 将每个 token 映射为对应的词表索引，并包裹为二维张量 (batch_size=1, seq_len)
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def get_sinusoid_encoding_table(n_position, d_model):
    """
    生成正弦位置编码表
    位置编码使用不同频率的正弦/余弦函数，使模型能够感知 token 在序列中的绝对位置
    偶数维度使用 sin，奇数维度使用 cos，波长从 2π 到 10000*2π 指数级增长
    :param n_position: 最大序列长度（位置数）
    :param d_model: 嵌入维度
    :return: 形状为 (n_position, d_model) 的位置编码张量
    """
    def cal_angle(position, hid_idx):
        # 计算位置 position 在维度 hid_idx 处的相位角
        # 公式: position / 10000^(2*(hid_idx//2)/d_model)
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

    def get_posi_angle_vec(position):
        # 为给定位置生成所有 d_model 维度的角度值
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    # 构建完整的位置编码表: (n_position, d_model)
    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    # 偶数维度应用 sin 函数 (对应维度 0, 2, 4, ..., 2i)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    # 奇数维度应用 cos 函数 (对应维度 1, 3, 5, ..., 2i+1)
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    """
    生成填充掩码 (Padding Mask)，用于遮蔽 PAD token 对应的注意力权重
    在自注意力或交叉注意力中，PAD 位置不应参与计算，因此将其注意力分数设为极小值
    :param seq_q: Query 序列, 形状 (batch_size, len_q)
    :param seq_k: Key 序列, 形状 (batch_size, len_k)
    :return: 布尔掩码, 形状 (batch_size, len_q, len_k)，值为 True 的位置表示需要遮蔽
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # seq_k.data.eq(0) 判断哪些位置是 PAD (索引为0), 得到 (batch_size, len_k) 的布尔张量
    # unsqueeze(1) 在第1维插入维度，变为 (batch_size, 1, len_k)，便于后续广播扩展
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # 沿 len_q 维度扩展，使每个 Query 位置对 Key 序列的 PAD 位置都进行遮蔽
    # 最终形状: (batch_size, len_q, len_k)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequent_mask(seq):
    """
    生成因果掩码 (Causal Mask / Subsequent Mask)，用于解码器自注意力
    构造上三角矩阵，遮蔽未来位置的信息，确保解码器在生成第 t 个 token 时
    只能看到前 t-1 个已生成的 token，保证自回归生成的正确性
    :param seq: 目标序列, 形状 (batch_size, seq_len)
    :return: 上三角掩码, 形状 (batch_size, seq_len, seq_len)，上三角部分为 1 (需要遮蔽)
    """
    # 掩码形状: (batch_size, seq_len, seq_len)
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # np.triu 提取上三角部分 (k=1 表示不包含对角线), 上三角置 1，其余为 0
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    # 转换为字节张量 (byte tensor)，值为 0/1
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力模块
    计算公式: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    缩放因子 1/sqrt(d_k) 用于防止 d_k 较大时点积结果过大导致 softmax 梯度消失
    """
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: Query 张量, 形状 (batch_size, n_heads, len_q, d_k)
        :param K: Key 张量, 形状 (batch_size, n_heads, len_k, d_k)
        :param V: Value 张量, 形状 (batch_size, n_heads, len_k, d_v)
        :param attn_mask: 注意力掩码, 形状 (batch_size, n_heads, len_q, len_k)
        :return: context 上下文向量 (batch_size, n_heads, len_q, d_v)
        : attn 注意力权重 (batch_size, n_heads, len_q, len_k)
        """
        # Q 与 K 的转置相乘得到注意力分数
        # K.transpose(-1, -2) 交换最后两维: (batch_size, n_heads, d_k, len_k)
        # 结果 scores 形状: (batch_size, n_heads, len_q, len_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 将掩码位置 (值为 True) 的分数填充为极小负数 (-1e9)
        # 这样在后续 softmax 时，这些位置的权重将趋近于 0
        scores.masked_fill_(attn_mask, -1e9)
        # 沿着最后一个维度 (len_k) 应用 softmax，使每个 Query 对所有 Key 的注意力权重和为 1
        attn = nn.Softmax(dim=-1)(scores)
        # 使用注意力权重对 Value 进行加权求和
        # attn: (batch_size, n_heads, len_q, len_k) × V: (batch_size, n_heads, len_k, d_v)
        # 结果 context 形状: (batch_size, n_heads, len_q, d_v)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块
    将 d_model 维度的嵌入空间分割为 n_heads 个子空间 (每个头 d_k = d_model / n_heads 维)
    每个头独立计算注意力，最后将所有头的输出拼接并投影回 d_model 维度
    这种设计使模型能够同时关注来自不同位置、不同表示子空间的信息
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # Q, K, V 的线性投影层: d_model → n_heads * d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        # 多头拼接后的输出投影层: n_heads * d_v → d_model
        self.linear = nn.Linear(n_heads * d_v, d_model)
        # 层归一化，用于残差连接之后
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: Query 输入, 形状 (batch_size, len_q, d_model)
        :param K: Key 输入, 形状 (batch_size, len_k, d_model)
        :param V: Value 输入, 形状 (batch_size, len_k, d_model)
        :param attn_mask: 注意力掩码, 形状 (batch_size, len_q, len_k)
        :return: 层归一化后的多头注意力输出 (batch_size, len_q, d_model)
                以及注意力权重 (batch_size, n_heads, len_q, len_k)
        """
        # 保存残差连接项，并获取 batch_size 用于后续维度变换
        residual, batch_size = Q, Q.size(0)

        # 维度变换流程: (B, S, D) → 线性投影 → (B, S, H*d_k) → view 分割 → (B, S, H, d_k) → transpose → (B, H, S, d_k)
        # 这样设计是为了让每个 head 的注意力计算可以并行进行
        # q_s: (batch_size, n_heads, len_q, d_k), 每个头独立处理 len_q 个 Query
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # k_s: (batch_size, n_heads, len_k, d_k), 每个头独立处理 len_k 个 Key
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # v_s: (batch_size, n_heads, len_k, d_v), 每个头独立处理 len_k 个 Value
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 掩码需要在 head 维度上复制，使其与 q_s/k_s 的维度对齐
        # attn_mask: (batch_size, 1, len_q, len_k) → repeat → (batch_size, n_heads, len_q, len_k)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 将分割后的 Q, K, V 送入缩放点积注意力，并行计算所有头的注意力
        # context: (batch_size, n_heads, len_q, d_v)
        # attn: (batch_size, n_heads, len_q, len_k)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # 将多头输出拼接回原始序列格式
        # transpose(1, 2): (B, H, S, d_v) → (B, S, H, d_v)
        # contiguous(): 确保内存连续，以便后续 view 操作正确
        # view(batch_size, -1, n_heads * d_v): 合并 head 维度，得到 (B, S, H*d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # 线性投影将拼接后的向量映射回 d_model 维度
        output = self.linear(context)
        # 残差连接 + 层归一化: LayerNorm(output + residual)，稳定训练并加速收敛
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    """
    位置级前馈网络 (Position-wise Feed-Forward Network)
    对序列中每个位置的向量独立进行相同的变换: d_model → d_ff → d_model
    使用 Conv1d(kernel_size=1) 等效于在每个位置应用全连接层，
    利用卷积的通道操作高效实现位置独立的逐点变换
    """
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # 第一层 Conv1d: 将通道数从 d_model 扩展至 d_ff (通常 d_ff = 4 * d_model)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # 第二层 Conv1d: 将通道数从 d_ff 压缩回 d_model
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 层归一化，用于残差连接之后
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        """
        :param inputs: 输入张量, 形状 (batch_size, seq_len, d_model)
        :return: 前馈网络输出, 形状 (batch_size, seq_len, d_model)
        """
        # 保存残差连接项
        residual = inputs

        # Conv1d 期望输入形状为 (batch_size, channels, seq_len)
        # 因此先 transpose(1, 2) 将序列长度维度与通道维度交换: (B, D, S)
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        # 第二层卷积后再次 transpose(1, 2) 恢复为 (B, S, D) 形状
        output = self.conv2(output).transpose(1, 2)
        # 残差连接 + 层归一化
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    """
    编码器层模块
    每个编码器层包含两个子层: 多头自注意力 + 位置级前馈网络
    每个子层后都跟随残差连接和层归一化
    """
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # 多头自注意力子层，编码器中 Q=K=V 均来自同一输入 (自注意力)
        self.enc_self_attn = MultiHeadAttention()
        # 位置级前馈网络子层
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: 编码器输入, 形状 (batch_size, src_len, d_model)
        :param enc_self_attn_mask: 填充掩码, 形状 (batch_size, src_len, src_len)
        :return: 编码器层输出 (batch_size, src_len, d_model) 和自注意力权重
        """
        # 自注意力: Q=K=V=enc_inputs，编码器直接关注输入序列内部的所有位置
        # 编码器输入的 Q, K, V 相同，因此是纯粹的自注意力机制
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # 前馈网络: 对每个位置的表示独立进行非线性变换
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    """
    解码器层模块
    每个解码器层包含三个子层: 掩蔽自注意力 + 交叉注意力 + 位置级前馈网络
    """
    def __init__(self):
        super(DecoderLayer, self).__init__()
        # 掩蔽自注意力子层，仅关注已生成的目标序列部分
        self.dec_self_attn = MultiHeadAttention()
        # 交叉注意力子层，Query 来自解码器，Key/Value 来自编码器输出
        self.dec_enc_attn = MultiHeadAttention()
        # 位置级前馈网络子层
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: 解码器输入, 形状 (batch_size, tgt_len, d_model)
        :param enc_outputs: 编码器输出, 形状 (batch_size, src_len, d_model)
        :param dec_self_attn_mask: 解码器自注意力掩码 (填充+因果), 形状 (batch_size, tgt_len, tgt_len)
        :param dec_enc_attn_mask: 交叉注意力掩码, 形状 (batch_size, tgt_len, src_len)
        :return: 解码器层输出及两种注意力权重
        """
        # 第一子层: 掩蔽自注意力，Q=K=V=dec_inputs
        # dec_self_attn_mask 同时包含填充掩码和因果掩码，防止看到 PAD 和未来位置
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # 第二子层: 交叉注意力，Query 为解码器输出，Key/Value 为编码器输出
        # 这一步将编码器的源语言信息注入到解码器中
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        # 第三子层: 前馈网络
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    """
    完整编码器
    由词嵌入层、位置编码层和 N 个编码器层堆叠组成
    将源序列转换为一组上下文感知的连续表示
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # 词嵌入层: 将离散 token 索引映射为连续向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 位置编码层: 使用预计算的正弦编码，freeze=True 表示不参与梯度更新
        # src_len+1 是为了包含一个额外的位置 (处理位置索引从1开始的情况)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len + 1, d_model), freeze=True)
        # 堆叠 n_layers 个编码器层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        :param enc_inputs: 源序列 token 索引, 形状 (batch_size, src_len)
        :return: 编码器最终输出 (batch_size, src_len, d_model) 及各层自注意力权重
        """
        # 词嵌入 + 位置编码: 将离散索引转为连续表示并注入位置信息
        # 此处使用硬编码的位置索引 [1,2,3,4,0]，0 对应 PAD 位置
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(torch.LongTensor([[1, 2, 3, 4, 0]]))

        # 生成填充掩码，确保 PAD 位置不参与注意力计算
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)

        enc_self_attns = []
        # 逐层通过编码器层，每层都进行自注意力和前馈变换
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)  # 收集各层的注意力权重用于可视化
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    """
    完整解码器
    由词嵌入层、位置编码层和 N 个解码器层堆叠组成
    自回归地生成目标序列，同时关注编码器的源序列表示
    """
    def __init__(self):
        super(Decoder, self).__init__()
        # 目标语言词嵌入层
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        # 目标语言位置编码层
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len + 1, d_model), freeze=True)
        # 堆叠 n_layers 个解码器层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        :param dec_inputs: 目标序列 token 索引 (含起始符S), 形状 (batch_size, tgt_len)
        :param enc_inputs: 源序列 token 索引, 形状 (batch_size, src_len)
        :param enc_outputs: 编码器输出, 形状 (batch_size, src_len, d_model)
        :return: 解码器最终输出及两种注意力权重
        """
        # 词嵌入 + 位置编码
        # 此处使用硬编码的位置索引 [5,1,2,3,4]，5 对应起始符 S 的位置编码索引
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5, 1, 2, 3, 4]]))

        # 生成解码器自注意力掩码: 结合填充掩码和因果掩码
        # 填充掩码: 遮蔽 PAD 位置
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        # 因果掩码: 遮蔽未来位置，确保自回归生成
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # 两种掩码相加后取 > 0，任一掩码为 1 的位置最终都会被遮蔽
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # 生成交叉注意力掩码: 解码器关注编码器时的填充掩码
        # 注意这里是 dec_inputs 和 enc_inputs，形状为 (batch_size, tgt_len, src_len)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        # 逐层通过解码器层
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(
                dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask
            )
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    """
    Transformer 模型
    完整的编码器-解码器架构，用于序列到序列的转换任务
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # 输出投影层: 将解码器的隐藏状态映射为目标词表上的 logits
        # bias=False 因为后续 softmax 不受偏置影响
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        :param enc_inputs: 源序列, 形状 (batch_size, src_len)
        :param dec_inputs: 目标输入序列, 形状 (batch_size, tgt_len)
        :return: 展平后的 logits (用于损失计算) 及各层注意力权重
        """
        # 编码器前向传播
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # 解码器前向传播，接收编码器输出用于交叉注意力
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # 投影到目标词汇空间: (batch_size, tgt_len, tgt_vocab_size)
        dec_logits = self.projection(dec_outputs)
        # 展平为 (batch_size * tgt_len, tgt_vocab_size)，以便与展平后的标签计算交叉熵损失
        # CrossEntropyLoss 期望输入为 (N, C) 形状，标签为 (N,) 形状
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns


def showgraph(attn):
    """
    可视化注意力权重矩阵
    展示最后一个编码器/解码器层的第一个注意力头的权重分布
    :param attn: 注意力权重列表，每个元素形状为 (batch_size, n_heads, len_q, len_k)
    """
    # 取最后一层的注意力权重，移除 batch 维度，取第一个头
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()


if __name__ == '__main__':
    # 训练数据: [源序列(德语), 目标输入(含S), 目标标签(含E)]
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer 模型参数配置
    # 注意: 填充符 P 的索引必须为 0，因为掩码计算依赖于 eq(0) 判断

    # 源语言词表 (德语)
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    # 目标语言词表 (英语)
    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    # 构建索引到 token 的反向映射，用于将预测结果转换回文本
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5   # 源序列长度
    tgt_len = 5   # 目标序列长度

    # 模型超参数 (遵循原始论文 "Attention Is All You Need" 的 base 配置)
    d_model = 512        # 词嵌入维度 / 隐藏状态维度
    d_ff = 2048          # 前馈网络中间层维度 (4 * d_model)
    d_k = d_v = 64       # 每个注意力头的键/值维度 (d_model / n_heads = 512 / 8 = 64)
    n_layers = 6         # 编码器/解码器的堆叠层数
    n_heads = 8          # 多头注意力的头数

    # 实例化模型
    model = Transformer()

    # 损失函数: 交叉熵损失 (内部包含 softmax)
    criterion = nn.CrossEntropyLoss()
    # 优化器: Adam, 学习率 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 构建训练批次
    enc_inputs, dec_inputs, target_batch = make_batch(sentences)

    # 训练循环
    for epoch in range(20):
        optimizer.zero_grad()  # 清零梯度
        # 前向传播
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # 计算损失: outputs 形状 (batch_size*tgt_len, tgt_vocab_size)
        # target_batch 展平为 (batch_size*tgt_len,)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

    # 测试: 直接使用原始目标输入进行推理 (教师强制模式)
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    # 取每个位置上概率最大的词作为预测结果
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # 可视化注意力权重
    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)
