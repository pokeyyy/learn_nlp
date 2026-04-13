# 代码作者: Tae Hwan Jung @graykode
import torch
import torch.nn as nn
import torch.optim as optim

def make_batch():
    input_batch = []
    target_batch = []

    for sen in sentences:
        word = sen.split()  # 使用空格作为分词器
        input = [word_dict[n] for n in word[:-1]]  # 创建前n-1个词作为输入
        target = word_dict[word[-1]]  # 创建第n个词作为目标，我们通常称之为"因果语言模型"

        input_batch.append(input)
        target_batch.append(target)

    return input_batch, target_batch

# NNLM模型
class NNLM(nn.Module):
    def __init__(self):
        super(NNLM, self).__init__()
        self.C = nn.Embedding(n_class, m)  # 词向量嵌入层：将词汇索引映射到m维向量
        self.H = nn.Linear(n_step * m, n_hidden, bias=False)  # 隐藏层：(n_step*m) -> n_hidden
        self.d = nn.Parameter(torch.ones(n_hidden))  # 隐藏层偏置参数
        self.U = nn.Linear(n_hidden, n_class, bias=False)  # 输出层：n_hidden -> n_class
        self.W = nn.Linear(n_step * m, n_class, bias=False)  # 直连层（跳跃连接）：(n_step*m) -> n_class
        self.b = nn.Parameter(torch.ones(n_class))  # 输出层偏置参数

    def forward(self, X):
        X = self.C(X)  # 词向量映射：X: [batch_size, n_step, m]
        X = X.view(-1, n_step * m)  # 展平为向量：[batch_size, n_step * m]
        tanh = torch.tanh(self.d + self.H(X))  # 隐藏层激活（tanh）：[batch_size, n_hidden]
        output = self.b + self.W(X) + self.U(tanh)  # 最终输出：[batch_size, n_class]
        return output

if __name__ == '__main__':
    n_step = 2  # 步数，论文中的n-1
    n_hidden = 2  # 隐藏层大小，论文中的h
    m = 2  # 词向量维度，论文中的m

    sentences = ["i like dog", "i love coffee", "i hate milk"]

    # 构建词汇表
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}  # 词->索引映射
    number_dict = {i: w for i, w in enumerate(word_list)}  # 索引->词映射
    n_class = len(word_dict)  # 词汇表大小

    model = NNLM()

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

    input_batch, target_batch = make_batch()
    input_batch = torch.LongTensor(input_batch)  # 转换为张量
    target_batch = torch.LongTensor(target_batch)  # 转换为张量
    #训练阶段
    for epoch in range(5000):
        optimizer.zero_grad()  # 梯度清零
        output = model(input_batch)  # 前向传播

        # output: [batch_size, n_class], target_batch: [batch_size]
        loss = criterion(output, target_batch)  # 计算损失
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

    # 预测阶段
    predict = model(input_batch).data.max(1, keepdim=True)[1]  # 获取预测词索引

    # 测试：显示预测结果
    # Test
    print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])