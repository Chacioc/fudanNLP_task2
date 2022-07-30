import torch
import numpy as np


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    print_hi('PyCharm')
    np_data = np.arange(6).reshape((2, 3))
    torch_data = torch.from_numpy(np_data)
    tensor2array = torch_data.numpy()
    print(
        '\nnumpy array:', np_data,  # [[0 1 2], [3 4 5]]
        '\ntorch tensor:', torch_data,  # 0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
        '\ntensor to array:', tensor2array,  # [[0 1 2], [3 4 5]]
    )
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
