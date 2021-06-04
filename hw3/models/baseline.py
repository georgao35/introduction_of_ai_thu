import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gensim


class MLP(nn.Module):
    def __init__(self,
                 weight: torch.Tensor,
                 vocab: dict,
                 embed_size: int,
                 dropout=0.75,
                 out_channel: int = 7,
                 hidden_dim: int = 32,
                 max_len: int = 100):
        super(MLP, self).__init__()
        self.embed_dim = embed_size
        self.hidden_dim = hidden_dim
        self.type_dim = out_channel
        self.vocab = vocab
        self.max_len = max_len
        self.device = 'cpu'

        # self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding = nn.Embedding(weight.shape[0], embed_size)
        # self.embedding.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(embed_size * max_len, self.hidden_dim),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_dim, out_channel),
            # nn.Softmax(dim=1)
        )

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x_in) -> torch.Tensor:
        """

        :param x_in: [[str]] with len of N, will get [N * len * vec_len] from embed
        :return: the score for each label
        """
        x = self.embed(x_in)
        # print(x)
        x = x.view((x.shape[0], -1))
        output = self.mlp(x)
        # print(output)
        return F.softmax(output, dim=1).to(self.device)

    def predict(self, x_in) -> torch.Tensor:
        """
        Soft max the score into prediction
        :param x_in:
        :return:
        """
        output = self.forward(x_in)
        return torch.argmax(output, dim=1).to(self.device)

    def loss(self, predicted: torch.Tensor, answer: torch.Tensor):
        # loss = -1 * torch.sum(answer * torch.log(predicted + 1e-18)) / predicted.shape[0]
        return self.loss_function(predicted, torch.argmax(answer, dim=1))

    def embed(self, x_in: list) -> torch.Tensor:
        vocab = self.vocab
        embedding = self.embedding
        max_len = self.max_len
        embed_dim = self.embed_dim
        sentence_tensors = []
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
