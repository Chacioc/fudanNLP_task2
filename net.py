import torch
import torch.nn as nn
import torch.nn.functional as tnf


class CNN(torch.nn.Module):
    def __init__(self, weight):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(len(weight), 50, _weight=weight)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 100, (k, 50), padding=(k - 1, 0))
            for k in [2, 3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(100 * 4, 5)

    def conv_and_pool(self, x, conv):
        x = tnf.relu(conv(x).squeeze(3))
        x_max = tnf.max_pool1d(x, x.size(2)).squeeze(2)
        return x_max

    def forward(self, x):
        embed = self.embed(x).unsqueeze(1).type(torch.FloatTensor).cuda()
        conv_results = [self.conv_and_pool(embed, conv) for conv in self.convs]

        out = torch.cat(conv_results, 1)
        return self.fc(self.dropout(out))


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()