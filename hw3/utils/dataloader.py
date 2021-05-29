import csv
import string


class DataLoader:
    def __init__(self, data_dir='../data'):
        self.raws = []
        self.data_dir = data_dir
        self.stop_words = []
        self.sentences = []
        self.label = []
        self.segment_labels = []
        self.load_stop_words()

    def load_stop_words(self, filename='data/stopwords.txt'):
        """
        read stop words
        :param filename:
        :return:
        """
        words = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for word in lines:
                words.append(word.strip('\n'))
        self.stop_words = words
        return words

    def load_sentences(self, filename='data/isear_train.csv'):
        lines = csv.reader(open(filename, 'r'))
        raws = [(line[2], line[1]) for line in lines]
        sentences = [raw[0] for raw in raws]
        labels = [raw[1] for raw in raws]
        self.raws = raws
        return sentences, labels

    def load_words(self):
        """
        break sentence into words and delete stop&unkown words
        :return:
        """
        raws = self.raws
        special_characters = ''.join([string.punctuation, 'á', '谩'])
        segments = []
        segment_label = []
        for (sentence, label) in raws:
            words = [word.strip(special_characters) for word in sentence.split()]
            segment_tmp = []
            for word in words:
                if word not in self.stop_words and word != '':
                    segment_tmp.append(word.lower())
            segments.append(segment_tmp)
            segment_label.append((segment_tmp, label))
        self.segment_labels = segment_label
        return segments, segment_label
