# %%
# code by Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn

# S: 解码器输入起始符
# E: 解码器输出结束符
# P: 填充符号，当序列长度不足 n_step 时用于补齐

def make_batch():
    """
    构建训练批次数据
    返回:
        input_batch:  [batch_size, n_step, n_class]   编码器输入（One-Hot）
        output_batch: [batch_size, n_step+1, n_class] 解码器输入（以'S'开头，One-Hot）
        target_batch: [batch_size, n_step+1]          目标标签（原始索引，非One-Hot）
    """
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        # 将输入和输出序列统一填充至 n_step 长度
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        # 将字符转换为词表索引
        input = [num_dic[n] for n in seq[0]]
        # 解码器输入以 'S' 开头，后接填充后的目标序列
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 目标序列以 'E' 结尾，用于计算损失时与解码器输出对齐
        target = [num_dic[n] for n in (seq[1] + 'E')]

        # One-Hot 编码: 索引数组 → [seq_len, n_class] 形状的二维数组
        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)  # 目标标签不使用 One-Hot，直接存索引

    # 转换为 PyTorch 张量
    # input_batch:  [batch_size, n_step, n_class]
    # output_batch: [batch_size, n_step+1, n_class]
    # target_batch: [batch_size, n_step+1]
    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


def make_testbatch(input_word):
    """
    构建测试批次数据
    参数:
        input_word: 待翻译的输入单词
    返回:
        input_batch:  [1, n_step, n_class]   增加 batch 维度
        output_batch: [1, n_step+1, n_class] 解码器输入以'S'开头，后续全为'P'占位
    """
    input_batch, output_batch = [], []

    # 输入单词填充至 n_step 长度
    input_w = input_word + 'P' * (n_step - len(input_word))
    input = [num_dic[n] for n in input_w]
    # 解码器输入: 'S' 开头，后续 n_step 个位置用 'P' 填充
    # 推理阶段利用 RNN 的自回归特性，单次前向传播即可得到所有时间步输出
    output = [num_dic[n] for n in 'S' + 'P' * n_step]

    input_batch = np.eye(n_class)[input]
    output_batch = np.eye(n_class)[output]

    # unsqueeze(0) 在首维增加 batch 维度，使形状匹配模型输入要求
    # 训练时 batch_size=len(seq_data)，测试时 batch_size=1
    return torch.FloatTensor(input_batch).unsqueeze(0), torch.FloatTensor(output_batch).unsqueeze(0)


# 模型定义
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        # 编码器 RNN: 将输入序列编码为隐藏状态向量
        # input_size=n_class: 每个时间步输入维度等于词表大小（One-Hot 编码）
        # hidden_size=n_hidden: 隐藏层维度，决定信息表示能力
        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        # 解码器 RNN: 以编码器最终隐藏状态为初始状态，逐步生成输出序列
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        # 全连接层: 将解码器每个时间步的隐藏状态映射到词表空间，得到各词的概率
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        """
        前向传播
        参数:
            enc_input:  [batch_size, n_step, n_class]     编码器输入
            enc_hidden: [num_layers, batch_size, n_hidden] 初始隐藏状态
            dec_input:  [batch_size, n_step+1, n_class]    解码器输入（Teacher Forcing）
        返回:
            logits: [n_step+1, batch_size, n_class] 每个时间步的词表 logits
        """
        # PyTorch RNN 要求输入形状为 (seq_len, batch, input_size)
        # 原始数据为 (batch, seq_len, input_size)，需交换第 0 和第 1 维
        # 转置后: enc_input → [n_step, batch_size, n_class]
        enc_input = enc_input.transpose(0, 1)
        # 转置后: dec_input → [n_step+1, batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)

        # 编码器前向传播
        # enc_cell 输出:
        #   _ (忽略):  [n_step, batch_size, n_hidden]  每个时间步的隐藏状态
        #   enc_states: [1, batch_size, n_hidden]      最后一层的最终隐藏状态
        # enc_states 作为整个输入序列的语义压缩表示，后续传递给解码器
        _, enc_states = self.enc_cell(enc_input, enc_hidden)

        # 解码器前向传播（Teacher Forcing 模式）
        # dec_input 以 'S' 开头，包含完整真实目标序列，无需逐时间步自回归
        # outputs: [n_step+1, batch_size, n_hidden]  每个时间步的解码器隐藏状态
        outputs, _ = self.dec_cell(dec_input, enc_states)

        # 全连接层将隐藏状态映射到词表维度
        # model: [n_step+1, batch_size, n_class]     每个时间步对应词表的 logits
        model = self.fc(outputs)
        return model

if __name__ == '__main__':
    n_step = 5       # 序列最大长度（时间步数），不足则用 'P' 填充
    n_hidden = 128   # RNN 隐藏层维度

    # 词表: 特殊符号 S/E/P + 26 个小写字母
    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dic = {n: i for i, n in enumerate(char_arr)}
    # 训练数据: 输入-输出单词对
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

    n_class = len(num_dic)   # 词表大小
    batch_size = len(seq_data)  # 批次大小等于数据总量

    model = Seq2Seq()

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，内部会对 logits 做 Softmax
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    for epoch in range(5000):
        # 初始化隐藏状态: [num_layers * num_directions, batch_size, n_hidden]
        # 本实现使用单层单向 RNN，故第 0 维为 1
        hidden = torch.zeros(1, batch_size, n_hidden)

        optimizer.zero_grad()
        # 输入维度说明:
        #   input_batch:  [batch_size, n_step, n_class]    编码器输入 (One-Hot)
        #   output_batch: [batch_size, n_step+1, n_class]  解码器输入 (以'S'开头，多一个时间步)
        #   target_batch: [batch_size, n_step+1]           目标标签 (原始索引，非 One-Hot)
        output = model(input_batch, hidden, output_batch)
        # model 返回: [n_step+1, batch_size, n_class]
        # 转置回 batch 优先顺序，便于逐样本计算损失
        output = output.transpose(0, 1)  # → [batch_size, n_step+1, n_class]
        loss = 0
        for i in range(0, len(target_batch)):
            # 逐个样本计算交叉熵损失并累加
            # output[i]:   [n_step+1, n_class]  该样本所有时间步的 logits
            # target[i]:   [n_step+1]           该样本所有时间步的真实标签索引
            # CrossEntropyLoss 会对 logits 自动做 Softmax，并与标签索引计算损失
            loss += criterion(output[i], target_batch[i])
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # 测试推理函数
    def translate(word):
        """
        将输入单词翻译为目标单词
        参数:
            word: 输入单词字符串
        返回:
            预测的目标单词字符串（去除 'P' 填充符）
        """
        input_batch, output_batch = make_testbatch(word)

        # 测试时 batch_size=1，隐藏状态形状: [1, 1, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden)
        output = model(input_batch, hidden, output_batch)
        # output: [n_step+1, 1, n_class]  所有时间步的词表 logits

        # 在词表维度（dim=2）取最大值位置，得到预测的词索引
        # predict: [n_step+1, 1, 1]  keepdim=True 保持维度便于后续索引操作
        predict = output.data.max(2, keepdim=True)[1]
        # 将索引转换回字符: decoded 为二维数组，元素为字符
        decoded = [char_arr[i] for i in predict]
        # 找到结束符 'E' 的位置，截取之前的字符序列
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        # 去除填充符 'P'，得到最终翻译结果
        return translated.replace('P', '')

    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))