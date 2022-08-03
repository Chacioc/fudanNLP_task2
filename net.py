import torch
import torch.nn as nn


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 52, 50)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=56,  # n_filters
                kernel_size=(2, 50),  # filter size
            ),  # output shape (56, 51, 1)
            nn.Tanh(),  # activation
            nn.MaxPool2d((55, 1))    # output (56, 1, 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=56,  # n_filters
                kernel_size=(3, 50),  # filter size
            ),  # output shape (56, 50, 1)
            nn.ReLU(),  # activation
            nn.MaxPool2d((54, 1))  # output (56, 1, 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=56,  # n_filters
                kernel_size=(4, 50),  # filter size
            ),  # output shape (56, 50, 1)
            nn.ReLU(),  # activation
            nn.MaxPool2d((53, 1))  # output (56, 1, 1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=56,  # n_filters
                kernel_size=(5, 50),  # filter size
            ),  # output shape (56, 50, 1)
            nn.ReLU(),  # activation
            nn.MaxPool2d((52, 1))  # output (56, 1, 1)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(56 * 4, 5)

    def forward(self, x):
        # print(self.conv1(x).shape)
        x1 = self.conv1(x).squeeze()
        x2 = self.conv2(x).squeeze()
        x3 = self.conv3(x).squeeze()
        x4 = self.conv4(x).squeeze()
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.dropout(x)
        # print(x.shape)
        return self.fc(x)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()