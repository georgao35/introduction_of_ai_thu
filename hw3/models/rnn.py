import torch
import torch.nn.functional as F
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,
                 weight: torch.Tensor,
                 vocab: dict,
                 embed_dim: int,
                 rnn_hidden_dim: int,
                 mlp_hidden_dim: int,
                 num_layer: int,
                 out_size: int = 7,
                 dropout=0.5,
                 max_len=128,
                 rtype: str = 'LSTM',
                 da_hidden_dim: int = 128,
                 self_attention: bool = True):
        """
        RNN网络
        :param weight: 预训练词向量，可以选择为0
        :param vocab: 词汇的词典
        :param embed_dim: 词向量维度
        :param rnn_hidden_dim: rnn隐层维度
        :param mlp_hidden_dim: 全连接隐藏层
        :param num_layer: rnn层数
        :param out_size: 输出标签个数
        :param dropout: dropout概率
        :param max_len: 句子最长长度
        :param rtype: rnn网络种类，目前支持lstm
        :param da_hidden_dim: attention网络隐层维度
        :param self_attention: 是否加上self attention
        """
        super(RNN, self).__init__()

        self.device = 'cpu'
        self.embedding = nn.Embedding(weight.shape[0], embed_dim)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.vocab = vocab
        self.loss_function = nn.CrossEntropyLoss()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.da_hidden_dim = da_hidden_dim
        if rtype == 'LSTM':
            self.rnn = nn.LSTM(input_size=embed_dim, hidden_size=rnn_hidden_dim,
                               num_layers=num_layer, batch_first=True, bidirectional=True)
        else:
            print('unsupported rnn type')
        self.mlp = nn.Sequential(
            nn.Linear(rnn_hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_size)
        )
        self.attention = self_attention
        if self_attention:
            self.W1 = nn.Parameter(torch.randn(da_hidden_dim, rnn_hidden_dim * 2))
            self.W2 = nn.Parameter(torch.randn(1, da_hidden_dim))

    def forward(self, x_in):
        x_embed, length = self.embed(x_in)
        x = torch.nn.utils.rnn.pack_padded_sequence(x_embed, length, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)
        if self.attention:
            output = output.permute(1, 0, 2)
            attention = F.softmax(
                torch.matmul(self.W2, torch.tanh(torch.matmul(self.W1, output.transpose(1, 2)))),
                dim=2)
            output = torch.matmul(attention, output)
            # print(output.shape)
            output = output.view(output.shape[0], -1)
        outputs = self.mlp(output[0]) if not self.attention else self.mlp(output)
        labels = F.softmax(outputs, dim=1)
        return labels

    def loss(self, predicted: torch.Tensor, answer: torch.Tensor):
        return self.loss_function(predicted, torch.argmax(answer, dim=1))

    def predict(self, x_in):
        labels = self.forward(x_in)
        return torch.argmax(labels, dim=1)

    def embed(self, x_in: list):
        """

        :param x_in:
        :return: embed in shape [batch * len * embed]
        """
        embed_dim = self.embed_dim
        sentence_tensors = []
        length = []
        vocab = self.vocab
        embedding = self.embedding
        max_len = self.max_len
        cnt = 0
        for sentence in x_in:
            if len(sentence):
                length.append(len(sentence))
                cnt += 1
            else:
                length.append(1)
            embedding_vectors = torch.zeros(max_len, embed_dim)
            for index, word in enumerate(sentence):
                try:
                    query_id = torch.tensor(vocab[word].index).to(self.device)
                    embedding_vectors[index] = embedding(query_id)
                except KeyError:
                    pass
            sentence_tensors.append(embedding_vectors)
        return torch.stack(sentence_tensors).to(self.device), length
