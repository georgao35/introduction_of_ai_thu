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
                 rtype: str = 'LSTM',):
        super(RNN, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(weight)
        self.vocab = vocab
        self.loss = nn.CrossEntropyLoss(reduction='sum')
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
        x = self.embed(x_in)
        rnn_out, _ = self.rnn(x)
        outputs = self.mlp(rnn_out[-1])
        labels = F.softmax(outputs, dim=1)
        return labels

    def loss_function(self, predicted: torch.Tensor, answer: torch.Tensor):
        return self.loss(predicted, torch.argmax(answer, dim=1))

    def predict(self, x_in):
        labels = self.forward(x_in)
        return torch.argmax(labels, dim=1)

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
