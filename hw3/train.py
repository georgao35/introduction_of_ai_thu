import random
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import DataLoader
from utils.batch import Batch
from gensim.models import Word2Vec
from multiprocessing import cpu_count
from models.baseline import MLP
from models.cnn import CNN
from models.rnn import RNN
from sklearn.metrics import f1_score

vector_size = 100


def parse_args():
    return []


def load_corpus():
    loader = DataLoader()
    sentences, labels = loader.load_sentences()
    stop_words = loader.load_stop_words()
    segments, segment_label = loader.load_words()
    return sentences, segments, segment_label


def train_w2vec(segments):
    # sentences = [' '.join(segment) for segment in segments]
    model = Word2Vec(sentences=segments,
                     sg=0,
                     size=vector_size,
                     window=10,
                     min_count=1,
                     sample=1e-3,
                     iter=5000,
                     hs=0,
                     workers=cpu_count())
    # print(model.wv.vocab)
    saving_model_path = 'data/train.model'
    model.save(saving_model_path)
    for s_word, f_sim in model.wv.most_similar('brother'):
        print(s_word, f_sim)
    return model


def load_w2vec(path: str = 'data/train.model'):
    model = Word2Vec.load(path)
    # wv = model.wv
    # for s_word, f_sim in wv.most_similar('preferred'):
    #     print(s_word, f_sim)
    return model


def init_network(model_type: str, gensim_model: Word2Vec, cnn_conv=None):
    """

    :param cnn_conv: kernel sizes for conv layer in conv
    :param model_type: type of chosen neural network
    :type gensim_model: Word2Vec
    """
    weight = torch.FloatTensor(gensim_model.wv.vectors)
    vocab = gensim_model.wv.vocab
    if model_type == 'MLP':
        network = MLP(weight, vocab, vector_size, hidden_dim=64)
    elif model_type == 'CNN':
        network = CNN(weight, vocab, vector_size, kernel_sizes=cnn_conv, hidden_channel=128, hidden_dim=40)
    elif model_type == 'RNN':
        network = RNN(weight, vocab, vector_size, rnn_hidden_dim=128, mlp_hidden_dim=128, num_layer=1)
    else:
        network = nn.Module()
    return network


def get_label_tensor(label: str) -> torch.Tensor:
    result = torch.zeros(7)
    index = 0
    if label == 'anger':
        index = 0
    elif label == 'disgust':
        index = 1
    elif label == 'fear':
        index = 2
    elif label == 'guilt':
        index = 3
    elif label == 'joy':
        index = 4
    elif label == 'sadness':
        index = 5
    elif label == 'shame':
        index = 6
    result[index] = 1
    return result


def train(network: nn.Module, batch_size: int, epochs: int, seg_labels: list):
    if torch.cuda.is_available():
        network = network.cuda()
        network.device = 'cuda'
    optim = torch.optim.Adam(network.parameters(), lr=1e-4)
    for i in range(epochs):
        network.train()
        batches = Batch(seg_labels, batch_size)
        for j, batch in enumerate(batches):
            sentences = []
            answers_list = []
            for (sample, label) in batch:
                sentences.append(sample)
                answers_list.append(get_label_tensor(label))
            answer = torch.stack(answers_list, dim=0).to(network.device)
            output = network.forward(sentences)
            # print(output)
            loss = network.loss(output, answer)
            optim.zero_grad()
            loss.backward()
            optim.step()
            accuracy, macro, micro = get_metric(torch.argmax(output, dim=1), torch.argmax(answer, dim=1))
            writer.add_scalar('train/loss', loss, i*batches.len + j)
            writer.add_scalar('train/accuracy', accuracy, i*batches.len + j)
            writer.add_scalar('train/macro', macro, i*batches.len + j)
            writer.add_scalar('train/micro', micro, i*batches.len + j)
        print("epoch%d" % i)
        evaluate(network)


def get_max_len(segments):
    max_len = 0
    for segment in segments:
        if len(segment) > max_len:
            max_len = len(segment)
    return max_len


def evaluate(network: nn.Module):
    eval_loader = DataLoader()
    eval_loader.load_stop_words()
    eval_loader.load_sentences(filename='data/isear_valid.csv')
    segments, segments_with_labels = eval_loader.load_words()
    batch = segments_with_labels
    sentences = []
    answers_list = []
    for (sample, label) in batch:
        sentences.append(sample)
        answers_list.append(get_label_tensor(label))
    answer = torch.argmax(torch.stack(answers_list, dim=0), dim=1)
    output = network.predict(sentences)
    print(get_metric(output, answer))


def get_metric(predicted: torch.Tensor, answer: torch.Tensor):
    """
    input tensors are of type [N * Labels], N is batch size and each label is id of corresponding label
    :param predicted: predicted labels
    :param answer: actual labels
    :return:
    """
    accuracy = torch.mean(torch.eq(answer.cpu(), predicted.cpu()).type(torch.FloatTensor))
    macro = f1_score(answer.cpu(), predicted.cpu(), average='macro')
    micro = f1_score(answer.cpu(), predicted.cpu(), average='micro')
    return accuracy, macro, micro


if __name__ == '__main__':
    args = parse_args()
    sentences_, segments_, segment_labels = load_corpus()
    # model = train_w2vec(segments_)
    model = load_w2vec()
    vector_size = model.wv.vector_size

    network_type = 'MLP'
    network = init_network(network_type, model, [3, 4, 5])

    log_dir_path = 'logs/%d%s' % (time.time(), network_type)
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)
    writer = SummaryWriter(log_dir_path)

    train(network, 64, 200, segment_labels)
    torch.save(network.state_dict(), 'models/saved_' + network_type + '2')
    evaluate(network)
