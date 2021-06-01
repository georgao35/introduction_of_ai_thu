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
                 self_attention: bool = False):
        super(RNN, self).__init__()

        self.device = 'cpu'
        self.embedding = nn.Embedding(weight.shape[0], embed_dim)
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.vocab = vocab
        self.loss_function = nn.CrossEntropyLoss()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mlp_hidden_dim = mlp_hidden_dim
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

    def forward(self, x_in):
        x_embed, length = self.embed(x_in)
        x = torch.nn.utils.rnn.pack_padded_sequence(x_embed, length, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(x)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out)
        outputs = self.mlp(output[0])
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
