import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    """创建训练批次数据"""
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # 空格分词
        input = [word_dict[n] for n in word[:-1]]  # 创建前n-1个词作为输入
        target = word_dict[word[-1]]  # 创建第n个词作为目标，这被称为因果语言模型

        input_batch.append(np.eye(n_class)[input])  # One-hot编码
        target_batch.append(target)

    return input_batch, target_batch  # 返回输入和目标批次

class TextRNN(nn.Module):
    """基于RNN的文本分类模型"""
    def __init__(self):
        super(TextRNN, self).__init__()
        # RNN层：输入维度=词汇表大小，隐层维度=隐藏单元数
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        # 线性变换层：将隐层输出映射到类别空间
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        # 偏置项
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        """前向传播
        
        参数:
            hidden: 初始隐状态 [num_layers * num_directions, batch_size, n_hidden]
            X: 输入张量 [batch_size, n_step, n_class]
            
        返回:
            model: 预测输出 [batch_size, n_class]
        """
        # 转置输入张量以满足RNN的输入格式要求
        X = X.transpose(0, 1)  # X: [n_step, batch_size, n_class]
        # RNN处理序列数据
        outputs, hidden = self.rnn(X, hidden)
        # outputs: [n_step, batch_size, num_directions(=1) * n_hidden]
        # hidden: [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # 取最后一个时间步的输出作为整个序列的特征表示
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        # 线性变换将隐层输出映射到词汇表大小
        model = self.W(outputs) + self.b  # model: [batch_size, n_class]
        return model

if __name__ == '__main__':
    # ========== 超参数配置 ==========
    n_step = 2  # RNN步数（序列长度）
    n_hidden = 5  # 隐层单元数

    # 训练数据
    sentences = ["i like dog", "i love coffee", "i hate milk"]

    # ========== 数据预处理 ==========
    # 构建词汇表
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))  # 去重
    # 建立词到索引的映射
    word_dict = {w: i for i, w in enumerate(word_list)}
    # 建立索引到词的反向映射
    number_dict = {i: w for i, w in enumerate(word_list)}
    n_class = len(word_dict)  # 词汇表大小
    batch_size = len(sentences)  # 批处理大小

    # ========== 模型初始化 ==========
    model = TextRNN()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    # ========== 数据准备 ==========
    input_batch, target_batch = make_batch()
    input_batch = torch.FloatTensor(input_batch)  # 转换为浮点张量
    target_batch = torch.LongTensor(target_batch)  # 转换为长整型张量

    # ========== 训练阶段 ==========
    for epoch in range(5000):
        # 梯度清零
        optimizer.zero_grad()

        # 初始化隐状态 [num_layers * num_directions, batch, hidden_size]
        hidden = torch.zeros(1, batch_size, n_hidden)
        # input_batch: [batch_size, n_step, n_class]
        # 前向传播
        output = model(hidden, input_batch)

        # output: [batch_size, n_class], target_batch: [batch_size]
        # 计算损失
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('第%d个epoch, 损失值 =' % (epoch + 1), '{:.6f}'.format(loss))

        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

    # ========== 预测阶段 ==========
    input = [sen.split()[:2] for sen in sentences]

    # 初始化隐状态进行预测
    hidden = torch.zeros(1, batch_size, n_hidden)
    # 获取模型预测结果（选择概率最高的类别）
    predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
    # 打印预测结果
    print('输入:', [sen.split()[:2] for sen in sentences], '预测:', [number_dict[n.item()] for n in predict.squeeze()])