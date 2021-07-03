import os, sys
import torch
from utils.dataloader import DataLoader
from models.rnn import RNN
from models.cnn import CNN
from models.baseline import MLP
from train import load_w2vec, evaluate


if __name__ == '__main__':
    model = load_w2vec('data/train1.model')
    vector_size = model.wv.vector_size

    test_network = 'RNN'
    file_path = 'models/RNN1'
    if test_network == '':
        print('Please specify a network')
        network = None
    elif test_network == 'CNN':
        network = CNN(model.wv.vectors, model.wv.vocab, vector_size,
                      kernel_sizes=[3, 4], hidden_channel=128, hidden_dim=128)
        network.load_state_dict(torch.load(file_path))
    elif test_network == 'MLP':
        network = MLP(model.wv.vectors, model.wv.vocab, vector_size, hidden_dim=128)
        network.load_state_dict(torch.load(file_path))
    elif test_network == 'RNN':
        network = RNN(model.wv.vectors, model.wv.vocab, vector_size, rnn_hidden_dim=128, mlp_hidden_dim=128,
                      num_layer=1, self_attention=False)
        network.load_state_dict(torch.load(file_path))
    else:
        print('got wrong nn type')
        network = None
    evaluate(network, 'data/isear_test.csv')
