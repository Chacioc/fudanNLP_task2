import torch
import torch.utils.data as tud
import torch.nn as nn
import pandas as pd
from net import CNN
from word_embeding import GloVe
from torchtext.vocab import Vectors

GLOVE_DATA_PATH = './data/glove.6B.50d.txt'
TRAIN_DATA_PATH = './data/train.tsv'
TEST_DATA_PATH = './data/test.tsv'
ANS_DATA_PATH = './data/ans.csv'
BATCH_SIZE = 64
EPOCH = 20
LR = 0.01

if __name__ == '__main__':
    glv = GloVe(GLOVE_DATA_PATH)
    train_df = pd.read_csv(TRAIN_DATA_PATH, sep='\t')
    x_data, y_data = train_df["Phrase"].values, train_df["Sentiment"].values
    x_matrix = glv.get_matrix(x_data)
    x_tensor = torch.from_numpy(x_matrix)
    y_tensor = torch.from_numpy(y_data)
    torch_dataset = tud.TensorDataset(x_tensor, y_tensor)
    loader = tud.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )

    cnn = CNN()
    cnn.cuda()
    # print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    # print("yes")
    cnn.train()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
            # print(b_x.shape)
            b_x, b_y = b_x.cuda(), b_y.cuda()
            b_x = b_x.unsqueeze(dim=1).type(torch.FloatTensor).cuda()
            # print(b_x.shape)
            output = cnn.forward(b_x)
            loss = loss_func(output, b_y).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cnn.eval()
    train_df = pd.read_csv(TEST_DATA_PATH, sep='\t')
    x_data, y_data = train_df["PhraseId"].values, train_df["Phrase"].values
    y_matrix = glv.get_matrix(y_data)
    y_tensor = torch.from_numpy(y_matrix).cuda()
    # print(y_tensor, '\n', y_tensor.shape)
    res = cnn(y_tensor.unsqueeze(dim=1).type(torch.FloatTensor).cuda())
    # print(res)
    res = torch.argmax(res, dim=1).cuda()
    # print(res)
    dataframe = pd.DataFrame({'PhraseId': x_data, 'Sentiment': res.cpu().detach().numpy()})
    dataframe.to_csv(ANS_DATA_PATH, index=False, sep=',')

