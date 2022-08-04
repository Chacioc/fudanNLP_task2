import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class GloVe:
    def __init__(self, path):
        temp = [[0] * 50]
        self.id = {}
        self.num = 0
        with open(path, 'r', encoding="utf-8") as ifs:
            for line in ifs.readlines():
                line = line.strip().split()
                word = line[0]
                vector = np.array(line[1:]).astype(float)
                temp.append(vector)
                self.num += 1
                self.id[word] = self.num
        self.weight = torch.from_numpy(np.array(temp))

    def get_id(self, word):
        if word in self.id:
            return self.id[word]
        return 0

    def get_matrix(self, sent_list):
        feature = []
        for i, sent in enumerate(sent_list):
            sent = sent.lower()
            words = sent.split()
            sent_feature = []
            for j, word in enumerate(words):
                sent_feature.append(self.get_id(word))
            feature.append(torch.LongTensor(sent_feature))
        return pad_sequence(feature, batch_first=True)

