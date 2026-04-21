# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer, https://github.com/dhlee347/pytorchic-bert
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 构造批次数据：确保小批量中 IsNext 和 NotNext 样本数量相等
def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        # 从句子列表中随机采样两个索引
        tokens_a_index, tokens_b_index= randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b= token_list[tokens_a_index], token_list[tokens_b_index]
        # 拼接输入序列：[CLS] + 句子A + [SEP] + 句子B + [SEP]
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        # 段落标识：句子A对应位置为0，句子B对应位置为1
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # 掩码语言建模（MLM）：随机掩盖15%的token
        n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 句子中15%的token作为候选
        # 候选掩码位置：排除[CLS]和[SEP]特殊token
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                        if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_maked_pos)  # 打乱候选位置
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%概率替换为[MASK]
                input_ids[pos] = word_dict['[MASK]']
            elif random() < 0.5:  # 10%概率替换为词汇表中的随机词（0.2 * 0.5 = 0.1）
                index = randint(0, vocab_size - 1)
                input_ids[pos] = word_dict[number_dict[index]]
            # 剩余10%概率保持原词不变

        # Zero Padding：将输入序列填充到最大长度
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # 对掩码位置和真实值进行填充，确保每个样本的掩码数量一致
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        # 控制正负样本比例：连续索引的句子对为正例（IsNext），否则为负例（NotNext）
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    return batch
# 数据预处理完成

# 生成注意力掩码：屏蔽padding位置，防止其参与注意力计算
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # seq_k中值为0的位置是PAD token，eq(0)返回布尔张量
    # unsqueeze(1)增加头的维度，用于后续广播到多个注意力头
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k，True表示需要屏蔽
    # 扩展为三维掩码：batch_size x len_q x len_k，每个查询位置使用相同的掩码
    return pad_attn_mask.expand(batch_size, len_q, len_k)

# GELU激活函数：高斯误差线性单元，BERT原始论文采用的激活函数
def gelu(x):
    "使用Hugging Face实现的GELU激活函数近似公式"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

# Embedding层：融合token嵌入、位置嵌入和段落类型嵌入
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token嵌入表
        self.pos_embed = nn.Embedding(maxlen, d_model)  # 绝对位置嵌入（可学习）
        self.seg_embed = nn.Embedding(n_segments, d_model)  # 段落（句子A/B）嵌入
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        # 生成位置索引 [0, 1, 2, ..., seq_len-1]
        pos = torch.arange(seq_len, dtype=torch.long)
        # 将位置索引扩展为与输入相同的批次维度：(seq_len,) -> (batch_size, seq_len)
        pos = pos.unsqueeze(0).expand_as(x)
        # 三种嵌入相加：token语义 + 绝对位置 + 段落归属
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

# 缩放点积注意力：计算 Q·K^T / √d_k 并用掩码屏蔽无效位置
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size x n_heads x len_q x d_k]
        # K: [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_k x d_v]
        # 计算注意力分数并缩放：除以√d_k防止点积过大导致梯度消失
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # 将掩码位置的分数填充为极小值（-1e9），使softmax后趋近于0
        scores.masked_fill_(attn_mask, -1e9)
        # 沿最后一个维度（key序列维度）进行softmax归一化
        attn = nn.Softmax(dim=-1)(scores)
        # 用注意力权重对V进行加权求和，得到上下文向量
        context = torch.matmul(attn, V)
        return context, attn

# 多头注意力机制：将Q/K/V投影到多个子空间并并行计算注意力
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 单线性层投影，后续通过view拆分多头，比创建多个独立Linear更高效
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)  # 保存残差用于后续残差连接

        # 维度变换流程：(B, S, D) → 线性投影 → (B, S, H*W) → view拆分 → (B, S, H, W) → 转置 → (B, H, S, W)
        # 转置目的是将heads维度移到第1维，使每个head的数据连续存储，便于批量矩阵乘法
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # [B x H x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # [B x H x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # [B x H x len_k x d_v]

        # 将注意力掩码从 [batch_size x len_q x len_k] 扩展为 [batch_size x n_heads x len_q x len_k]
        # unsqueeze(1)增加头维度，repeat沿头维度复制n_heads次
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # 计算缩放点积注意力，得到上下文向量和注意力权重
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # 上下文向量维度还原：(B, H, len_q, d_v) → 转置 → (B, len_q, H, d_v) → 拼接 → (B, len_q, H*d_v)
        # contiguous()确保转置后内存连续，view才能正确拼接
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        # 将多头拼接后的向量投影回原始模型维度d_model
        output = nn.Linear(n_heads * d_v, d_model)(context)
        # 残差连接 + LayerNorm：防止梯度消失，稳定训练
        return nn.LayerNorm(d_model)(output + residual), attn

# 位置前馈网络：对每个位置独立进行两次线性变换，中间使用GELU激活
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 升维：d_model → d_ff（通常4倍）
        self.fc2 = nn.Linear(d_ff, d_model)  # 降维：d_ff → d_model

    def forward(self, x):
        # 维度变化：(B, L, D) → (B, L, 4D) → GELU激活 → (B, L, 4D) → (B, L, D)
        # 逐位置独立变换，不跨序列维度交互
        return self.fc2(gelu(self.fc1(x)))

# Encoder层：由多头自注意力和位置前馈网络组成，各带残差连接和LayerNorm
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # 自注意力：Q=K=V均来自enc_inputs，使用同一输入的自关联注意力
        # enc_self_attn_mask屏蔽padding位置，使其不参与注意力计算
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # 前馈网络：逐位置进行非线性变换
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

# BERT主模型：Embedding + N层Encoder + 双任务输出头（MLM + NSP）
class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        # 堆叠n_layers个Encoder层
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        # NSP任务：对[CLS]位置输出进行池化分类
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        # MLM任务：对被掩码位置的隐藏状态进行变换后预测vocab
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)  # 二分类：IsNext / NotNext

        # MLM解码器：与token embedding共享权重，减少参数量
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight  # 权重绑定（Weight Tying）
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))  # 解码器偏置项

    def forward(self, input_ids, segment_ids, masked_pos):
        # 第一步：输入嵌入（token + position + segment）
        output = self.embedding(input_ids, segment_ids)
        # 生成padding注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        # 第二步：逐层Encoder处理
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # 此时 output: [batch_size, maxlen, d_model]

        # --- NSP任务头：基于[CLS] token进行句子对分类 ---
        # 取[CLS]位置（第一个token）的最终隐藏状态作为句子级表征
        h_pooled = self.activ1(self.fc(output[:, 0]))  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # [batch_size, 2] 二分类logits

        # --- MLM任务头：基于被掩码位置进行token预测 ---
        # 将masked_pos从 [B, M] 扩展为 [B, M, D]，用于gather操作
        # masked_pos[:, :, None] 增加最后一个维度，expand沿该维度复制d_model次
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        # 从output中按位置索引提取被掩码token对应的隐藏状态
        # torch.gather(dim=1)沿序列维度收集指定位置的向量
        h_masked = torch.gather(output, 1, masked_pos)  # [batch_size, max_pred, d_model]
        # 对提取的隐藏状态进行变换：Linear → GELU → LayerNorm
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        # 通过共享权重的解码器映射到词汇表维度，加上偏置得到预测logits
        logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, max_pred, vocab_size]

        return logits_lm, logits_clsf

if __name__ == '__main__':
    # BERT 超参数配置
    maxlen = 30  # 序列最大长度（含特殊token和padding）
    batch_size = 6  # 批次大小
    max_pred = 5  # 每个样本最多预测的掩码token数
    n_layers = 6  # Encoder层数
    n_heads = 12  # 多头注意力的头数
    d_model = 768  # 嵌入向量维度
    d_ff = 768 * 4  # 前馈网络隐藏层维度（4倍d_model）
    d_k = d_v = 64  # Q/K/V的维度，d_model / n_heads = 768 / 12 = 64
    n_segments = 2  # 段落类型数（句子A和句子B）

    # 示例文本数据
    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    # 文本预处理：转小写、去除标点符号、按行分割为句子列表
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
    # 构建词汇表：从所有句子中提取不重复的词
    word_list = list(set(" ".join(sentences).split()))
    # 词汇表词典：特殊token + 实际词汇
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    # 反向词典：用于将ID映射回词
    number_dict = {i: w for i, w in enumerate(word_dict)}
    vocab_size = len(word_dict)  # 词汇表大小

    # 将句子转换为token ID列表
    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    # 实例化模型、损失函数和优化器
    model = BERT()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 构造批次数据
    batch = make_batch()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

    # 训练循环
    for epoch in range(100):
        optimizer.zero_grad()
        # 前向传播：获取MLM和NSP的预测logits
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        # MLM损失：logits_lm维度[B, M, V]需转置为[B, V, M]以匹配CrossEntropyLoss的输入格式
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
        loss_lm = (loss_lm.float()).mean()
        # NSP损失：直接对比分类logits与标签
        loss_clsf = criterion(logits_clsf, isNext)
        # 总损失为两项之和
        loss = loss_lm + loss_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # 使用第一个样本进行预测，验证模型输出
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
    print(text)
    print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])

    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()  # 取词汇表维度最大值对应的ID
    print('masked tokens list : ',[pos.item() for pos in masked_tokens[0] if pos.item() != 0])
    print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('isNext : ', True if isNext else False)
    print('predict isNext : ',True if logits_clsf else False)
