import numpy as np


class GloVe:
    def __init__(self, path):
        self.glove = {}
        self.id = {}
        self.num = 0
        with open(path, 'r', encoding="utf-8") as ifs:
            for line in ifs.readlines():
                line = line.strip().split()
                word = line[0]
                vector = np.array(line[1:]).astype(float)
                self.glove[word] = vector
                self.num += 1
                self.id[word] = self.num

    def get_id(self, word):
        if word in self.id:
            return self.id[word]
        self.num += 1
        self.id[word] = self.num
        self.glove[word] = np.zeros(50)
        return self.num

    def get_vector(self, word):
        if word in self.glove:
            return self.glove[word]
        self.num += 1
        self.id[word] = self.num
        ans = self.glove[word] = np.zeros(50)
        return ans

    def get_matrix(self, sent_list):
        feature = np.zeros((len(sent_list), 56, 50))
        for i, sent in enumerate(sent_list):
            sent = sent.lower()
            sent = sent.replace(old='-', new=' ')
            words = sent.split()
            for j, word in enumerate(words):
                feature[i][j] = self.get_vector(word)
        return np.array(feature)

