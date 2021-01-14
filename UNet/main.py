import time
import torch
import numpy as np
import matplotlib.pyplot as plt


from utils import *
from network import *


def main():
    begin = time.time()
    dataset = ReadData()
    dataset.normalize()
    dataset.shuffle()
    train_data, train_label, test_data, test_label = dataset.split(
        boundary=71, Tensor=True)
    print("读取数据用时:{:.3f}s".format(time.time()-begin))

    model = U_Net(1, 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, loss_list, score_list = train(
        model, train_data, train_label, device, epochs=2)
    test_score_list = test(model, test_data, test_label, device)
    plt.scatter(np.arange(len(test_score_list)), test_score_list)
    plt.show()


if __name__ == '__main__':
    main()
