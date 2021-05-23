import torch
from train import *

sentences_, segments_, segment_labels = load_corpus()
model = load_w2vec()
mlp = init_network('MLP', model, cnn_conv=[3, 3, 4, 5])
tmp = segments_[10:19]
parameters = torch.load('models/saved_mlp')
mlp.load_state_dict(parameters)
evaluate(mlp)
# print(cnn(tmp), cnn(tmp).shape)
