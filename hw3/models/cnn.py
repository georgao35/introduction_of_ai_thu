import torch
import torch.nn.functional as F
import torch.nn as nn


def MaxPoolOvertime(x):
    return F.max_pool1d(x, kernel_size=x.shape[2])


class CNN(nn.Module):
    def __init__(self,
                 weight: torch.Tensor,
                 vocab: dict,
                 embed_size: int,
                 kernel_sizes=None,
                 out_size: int = 7,
                 max_len: int = 128,
                 hidden_channel: int = 32,
                 hidden_dim: int = 32,
                 dropout: float = 0.1):
        """
        CNN网络
        :param weight: 预训练词向量，可以选择为0
        :param vocab: 词汇的词典
        :param embed_size: 词向量维度
        :param kernel_sizes: 卷积核大小，多个卷积核组成列表
        :param out_size: 输出标签个数，为7
        :param max_len: 句子最大长度
        :param hidden_channel: cnn卷积核输出channel个数
        :param hidden_dim: 全连接网络隐含层维度
        :param dropout: dropout概率
        """
        super(CNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3]
        self.device = 'cpu'
        self.vocab = vocab
        self.embed_dim = embed_size
        self.dropout_rate = dropout
        self.hidden_channel = hidden_channel
        self.hidden_dim = hidden_dim
        self.feature_dim = out_size
        self.max_len = max_len
        self.loss_function = nn.CrossEntropyLoss()

        self.embedding = nn.Embedding(weight.shape[0], embed_size)
        # self.embedding.requires_grad_(False)
        self.conv = nn.ModuleList()
        self.conv_num = len(kernel_sizes)
        for kernel_size in kernel_sizes:
            self.conv.append(nn.Sequential(
                nn.Conv1d(embed_size, self.hidden_channel, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.mlp_network = nn.Sequential(
            nn.Linear(self.conv_num * self.hidden_channel, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, out_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x_in) -> torch.Tensor:
        """
        forward function for cnn network
        :param x_in: [[str]] with len of N, will get [N * len * vec_len] from embed
        :return: the score(softmax prob) for each label
        """
        x_vec = self.embed(x_in)
        batch_num = x_vec.shape[0]
        x_vec = x_vec.permute(0, 2, 1)
        cnn_output = torch.cat([MaxPoolOvertime(cnn(x_vec)).squeeze(2) for cnn in self.conv], dim=1).to(self.device)
        cnn_output = cnn_output.view(batch_num, -1)
        return self.mlp_network(cnn_output)

    def predict(self, x_in):
        """
        prediction function for cnn network
        :param x_in: [[str]] with len of N, will get [N * len * vec_len] from embed
        :return: the score(softmax prob) for each label
        """
        output = self.forward(x_in)
        return torch.argmax(output, dim=1).to(self.device)

    def loss(self, predicted: torch.Tensor, answer: torch.Tensor):
        return self.loss_function(predicted, torch.argmax(answer, dim=1))

    def embed(self, x_in: list) -> torch.Tensor:
        embed_dim = self.embed_dim
        sentence_tensors = []
        vocab = self.vocab
        embedding = self.embedding
        max_len = self.max_len
        for sentence in x_in:
            embedding_vectors = torch.zeros(max_len, embed_dim)
            for index, word in enumerate(sentence):
                try:
                    query_id = torch.tensor(vocab[word].index).to(self.device)
                    embedding_vectors[index] = embedding(query_id)
                except KeyError:
                    pass
            sentence_tensors.append(embedding_vectors)
        return torch.stack(sentence_tensors).to(self.device)
