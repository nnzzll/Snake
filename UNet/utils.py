
import os
import cv2
import glob
import copy
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from network import *


class ReadData:
    def __init__(self, data_path: str = os.getcwd()+"\\data", mask_path: str = os.getcwd()+"\\mask_int", data_shape: tuple = (512, 512)):
        '''
        使用OpenCV读取图片与标签存储为numpy数组
        图片数组的值域为0~255,标签数组为0,255的二值图
        '''
        self.data_list = glob.glob(data_path+"\\*")
        self.mask_list = glob.glob(mask_path+"\\*")
        if len(data_shape) == 2:
            self.image_data = np.zeros(
                [len(self.data_list), data_shape[0], data_shape[1]], dtype=np.uint8)
            self.mask_data = np.zeros(
                [len(self.mask_list), data_shape[0], data_shape[1]], dtype=np.uint8)
            for i in range(len(self.data_list)):
                self.image_data[i, :, :] = cv2.imread(self.data_list[i], 0)
                self.mask_data[i, :, :] = cv2.imread(self.mask_list[i], 0)

    def normalize(self):
        '''
        归一化图像数据与标签数据到(0,1)
        '''
        self.image_data = self.image_data/255.
        self.mask_data = self.mask_data/255.
        # self.image_data = self.image_data/np.max(self.image_data)
        # self.mask_data = self.mask_data/np.max(self.mask_data)

        return self.image_data, self.mask_data

    def shuffle(self, seed: int = 2021):
        '''
        打乱输入数据的第一维
        seed : 随机种子
        '''
        np.random.seed(seed)
        np.random.shuffle(self.image_data)
        np.random.seed(seed)
        np.random.shuffle(self.mask_data)
        return self.image_data, self.mask_data

    def split(self, ratio: float = 0.25,  boundary=None, Tensor=False):
        '''
        划分训练集与测试集
        ratio : 测试集所占比例
        boundary : 指定测试集数量
        Tensor : 是否转换成Torch.Tensor
        '''
        if not boundary:
            boundary = int(len(self.image_data)*ratio)
            test_data = self.image_data[:boundary, :, :]
            test_label = self.mask_data[:boundary, :, :]
            train_data = self.image_data[boundary:, :, :]
            train_label = self.mask_data[boundary:, :, :]
        else:
            test_data = self.image_data[:boundary, :, :]
            test_label = self.mask_data[:boundary, :, :]
            train_data = self.image_data[boundary:, :, :]
            train_label = self.mask_data[boundary:, :, :]

        if not Tensor:
            return train_data, train_label, test_data, test_label
        else:
            return torch.Tensor(train_data), torch.Tensor(train_label), torch.Tensor(test_data), torch.Tensor(test_label)


def train(model, data: torch.Tensor, mask: torch.Tensor, device: str, epochs: int = 50, batch_size=10, lr: float = 0.001, weight_decay: float = 5e-4):
    if len(data.size()) == 3:
        data = data.unsqueeze(1)
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(1)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    criterion = DiceLoss()
    loss_list = []
    score_list = []
    n_batch = len(data)//batch_size
    model.train()
    train_start_time = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        running_score = 0.0
        i = 0
        epoch_start_time = time.time()
        for step in range(n_batch):
            inputs = data[i:i+batch_size, :, :, :].to(device)
            labels = mask[i:i+batch_size, :, :, :].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            score = IOU(outputs.cpu().detach().numpy(),
                        labels.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_score += score
            i += batch_size
        print("Epoch: {}/{}\t Loss: {:.4f}\t Score: {:.4f}\t Time:{:.3f}s".format(epoch+1, epochs,
                                                                                  running_loss/step, running_score/step, time.time()-epoch_start_time))
        loss_list.append(running_loss/step)
        score_list.append(running_score/step)
    print("训练完成,用时:{:.3f}s".format(time.time()-train_start_time))
    return model, loss_list, score_list


def test(model, data: torch.Tensor, mask: torch.Tensor, device: str):
    if len(data.size()) == 3:
        data = data.unsqueeze(1)
    if len(mask.size()) == 3:
        mask = mask.unsqueeze(1)

    model.eval()
    with torch.no_grad():
        test_score_list = []
        test_start_time = time.time()
        for i in range(len(data)):
            inputs = data[i].unsqueeze(1).to(device)
            labels = mask[i].unsqueeze(1).to(device)
            outputs = model(inputs)
            score = IOU(outputs.detach().cpu().numpy(),
                        labels.detach().cpu().numpy())
            test_score_list.append(score)
    test_score_list = np.array(test_score_list, dtype=float)
    print("训练用时: {:.3f}s  \t  Average Score:{:.4f}".format(
        time.time()-test_start_time, np.mean(test_score_list)))
    return test_score_list


def IOU(pred, truth):
    pred = pred.reshape(-1)
    truth = truth.reshape(-1)
    intersection = (pred*truth).sum()
    total = (pred+truth).sum()
    union = total - intersection
    smooth = 1
    score = (intersection+smooth)/(union+smooth)
    return score


def plot_loss_and_score(loss, score):
    loss = np.array(loss, dtype=float)
    score = np.array(score, dtype=float)
    plt.plot(loss, label='Loss')
    plt.plot(score, label='Score')
    plt.legend(loc='best')
    plt.show()
