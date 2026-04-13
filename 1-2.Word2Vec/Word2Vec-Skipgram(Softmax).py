# 代码作者: Tae Hwan Jung @graykode
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def random_batch():
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[skip_grams[i][0]])  # 目标词
        random_labels.append(skip_grams[i][1])  # 上下文词

    return random_inputs, random_labels

# 模型
class Word2Vec(nn.Module):
    def __init__(self):
        super(Word2Vec, self).__init__()
        # W和WT不是转置关系
        self.W = nn.Linear(voc_size, embedding_size, bias=False) # 词汇表大小 > 嵌入维度 权重矩阵
        self.WT = nn.Linear(embedding_size, voc_size, bias=False) # 嵌入维度 > 词汇表大小 权重矩阵

    def forward(self, X):
        # X : [批大小, 词汇表大小]
        hidden_layer = self.W(X) # 隐层 : [批大小, 嵌入维度]
        output_layer = self.WT(hidden_layer) # 输出层 : [批大小, 词汇表大小]
        return output_layer

if __name__ == '__main__':
    batch_size = 2 # 小批量大小
    embedding_size = 2 # 嵌入维度

    sentences = ["apple banana fruit", "banana orange fruit", "orange banana fruit",
                 "dog cat animal", "cat monkey animal", "monkey dog animal"]

    word_sequence = " ".join(sentences).split()
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    voc_size = len(word_list)

    # 生成大小为1的skip-gram对
    skip_grams = []
    for i in range(1, len(word_sequence) - 1):
        target = word_dict[word_sequence[i]]
        context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]
        for w in context:
            skip_grams.append([target, w])

    model = Word2Vec()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    for epoch in range(5000):
        input_batch, target_batch = random_batch()
        input_batch = torch.Tensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        optimizer.zero_grad()
        output = model(input_batch)

        # 输出 : [批大小, 词汇表大小], 目标批 : [批大小] (LongTensor, 不是one-hot)
        loss = criterion(output, target_batch)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    for i, label in enumerate(word_list):
        W, WT = model.parameters()
        x, y = W[0][i].item(), W[1][i].item()
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
