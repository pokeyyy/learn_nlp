# %%
# code by Tae Hwan Jung @grayplate
# Reference : https://github.com/hunkim/PyTorchZeroToAll/blob/master/14_2_seq2seq_att.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# S: 解码器输入的起始标记
# E: 解码器输出的结束标记
# P: 当当前批次数据长度短于时间步数时，用于填充空白序列的占位符

def make_batch():
    """
    构建训练批次数据
    将源句子、解码输入句子转换为 one-hot 编码，将目标句子转换为词索引序列
    """
    # 源句子 one-hot 编码: [batch_size=1, n_step, n_class]
    input_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[0].split()]]]
    # 解码输入句子 one-hot 编码: [batch_size=1, n_step, n_class]
    output_batch = [np.eye(n_class)[[word_dict[n] for n in sentences[1].split()]]]
    # 目标句子词索引: [batch_size=1, n_step]
    target_batch = [[word_dict[n] for n in sentences[2].split()]]

    # 转换为 PyTorch 张量
    # input_batch/output_batch: FloatTensor, 形状 [1, n_step, n_class]
    # target_batch: LongTensor, 形状 [1, n_step]
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # 编码器 RNN: 输入维度为词表大小，隐藏层维度为 n_hidden
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        # 解码器 RNN: 输入维度为词表大小，隐藏层维度为 n_hidden
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)

        # 注意力计算所需的线性层: 将编码隐藏状态映射到注意力空间
        self.attn = nn.Linear(n_hidden, n_hidden)
        # 输出线性层: 将解码输出与上下文向量拼接后的维度 (n_hidden * 2) 映射到词表维度
        self.out = nn.Linear(n_hidden * 2, n_class)

    def forward(self, enc_inputs, hidden, dec_inputs):
        """
        前向传播
        Args:
            enc_inputs: 编码器输入, 形状 [batch_size, n_step, n_class]
            hidden: 初始隐藏状态, 形状 [num_layers, batch_size, n_hidden]
            dec_inputs: 解码器输入, 形状 [batch_size, n_step, n_class]
        Returns:
            model: 解码器输出 logits, 形状 [n_step, n_class]
            trained_attn: 记录的注意力权重矩阵, 形状 [n_step, n_step]
        """
        # 转置操作: 将 batch 维度置前，适配 PyTorch RNN 的输入格式要求
        # enc_inputs: [n_step, batch_size, n_class]
        enc_inputs = enc_inputs.transpose(0, 1)
        # dec_inputs: [n_step, batch_size, n_class]
        dec_inputs = dec_inputs.transpose(0, 1)

        # enc_outputs: 编码器所有时间步的隐藏状态, 形状 [n_step, batch_size, n_hidden]
        #             即注意力机制中的记忆矩阵 F
        # enc_hidden: 编码器最终隐藏状态, 形状 [num_layers, batch_size, n_hidden]
        enc_outputs, enc_hidden = self.enc_cell(enc_inputs, hidden)

        trained_attn = []
        # 将编码器最终隐藏状态作为解码器的初始隐藏状态
        hidden = enc_hidden
        n_step = len(dec_inputs)  # 解码时间步数
        # 预分配输出张量，用于累积每个时间步的预测结果
        model = torch.empty([n_step, 1, n_class])

        for i in range(n_step):  # 逐个时间步进行解码
            # 将当前时间步的解码输入从 [batch_size, n_class] 扩展为 [1, batch_size, n_class]
            # 以适配 RNN 的输入格式要求 (seq_len, batch, input_size)
            dec_output, hidden = self.dec_cell(dec_inputs[i].unsqueeze(0), hidden)
            # 计算当前解码隐藏状态对所有编码隐藏状态的注意力权重
            # attn_weights 形状: [batch_size=1, seq_len=1, enc_seq_len=n_step]
            attn_weights = self.get_att_weight(dec_output, enc_outputs)
            trained_attn.append(attn_weights.squeeze().data.numpy())

            # 上下文向量计算: 注意力权重与编码输出矩阵相乘
            # 维度推导: [1, 1, n_step] @ [1, n_step, n_hidden] = [1, 1, n_hidden]
            # enc_outputs.transpose(0, 1) 将形状从 [n_step, 1, n_hidden] 转为 [1, n_step, n_hidden]
            context = attn_weights.bmm(enc_outputs.transpose(0, 1))
            # 移除时间步维度，恢复为 [batch_size, n_hidden]
            dec_output = dec_output.squeeze(0)
            # 移除批次维度，恢复为 [1, n_hidden]
            context = context.squeeze(1)
            # 将解码输出和上下文向量在特征维度拼接，通过线性层生成词表 logits
            # 拼接后维度: [1, n_hidden * 2] -> 线性层 -> [1, n_class]
            model[i] = self.out(torch.cat((dec_output, context), 1))

        # 将输出从 [n_step, 1, n_class] 转置并压缩为 [n_step, n_class]
        return model.transpose(0, 1).squeeze(0), trained_attn

    def get_att_weight(self, dec_output, enc_outputs):
        """
        计算当前解码状态对所有编码状态的注意力权重分布
        Args:
            dec_output: 当前解码器 RNN 输出, 形状 [1, 1, n_hidden]
            enc_outputs: 编码器所有时间步隐藏状态, 形状 [n_step, 1, n_hidden]
        Returns:
            softmax 归一化后的注意力权重, 形状 [1, 1, n_step]
        """
        n_step = len(enc_outputs)
        # 初始化注意力分数张量
        attn_scores = torch.zeros(n_step)

        # 逐个计算当前解码状态与每个编码状态的注意力分数
        for i in range(n_step):
            attn_scores[i] = self.get_att_score(dec_output, enc_outputs[i])

        # 使用 softmax 将分数归一化为概率分布，并调整形状为 [1, 1, n_step]
        # view(1, 1, -1) 确保输出符合批次矩阵乘法的维度要求
        return F.softmax(attn_scores).view(1, 1, -1)

    def get_att_score(self, dec_output, enc_output):
        """
        计算单个解码状态与单个编码状态的注意力分数（点积形式）
        Args:
            dec_output: 当前解码器隐藏状态, 形状 [1, 1, n_hidden]
            enc_output: 单个编码器隐藏状态, 形状 [1, n_hidden]
        Returns:
            标量注意力分数
        """
        # 对编码隐藏状态进行线性变换，使其与解码状态处于同一语义空间
        # 变换后形状: [1, n_hidden]
        score = self.attn(enc_output)
        # 将两个展平后的向量做内积，得到标量注意力分数
        # dec_output.view(-1) 形状: [n_hidden]
        # score.view(-1) 形状: [n_hidden]
        return torch.dot(dec_output.view(-1), score.view(-1))

if __name__ == '__main__':
    n_step = 5  # 序列长度（RNN 时间步数）
    n_hidden = 128  # RNN 隐藏层维度

    # 训练数据: [源句子, 解码输入, 解码目标输出]
    # 源句子: 德语 "我想来杯啤酒" (P 为填充符)
    # 解码输入: 英语以 S 开头 (起始标记)
    # 解码目标: 英语以 E 结尾 (结束标记)
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # 构建词表
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    # 词到索引的映射
    word_dict = {w: i for i, w in enumerate(word_list)}
    # 索引到词的映射
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # 词表大小

    # 初始化隐藏状态: [num_layers, batch_size, n_hidden]
    hidden = torch.zeros(1, 1, n_hidden)

    model = Attention()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    # 训练循环
    for epoch in range(2000):
        optimizer.zero_grad()
        output, _ = model(input_batch, hidden, output_batch)

        # 计算交叉熵损失
        # output 形状: [n_step, n_class], target_batch.squeeze(0) 形状: [n_step]
        loss = criterion(output, target_batch.squeeze(0))
        if (epoch + 1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # 测试推理
    # 构造测试输入: 'SPPPP' 表示起始标记 + 填充符，模拟自回归生成初始状态
    test_batch = [np.eye(n_class)[[word_dict[n] for n in 'SPPPP']]]
    test_batch = torch.FloatTensor(test_batch)
    predict, trained_attn = model(input_batch, hidden, test_batch)
    # 取每个时间步 logits 最大值对应的索引作为预测结果
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    # 可视化注意力权重热力图
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(trained_attn, cmap='viridis')
    # 设置 x 轴标签为源句子 token
    ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
    # 设置 y 轴标签为目标句子 token
    ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()