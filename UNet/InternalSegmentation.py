import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import *
from network import *


def InternalSegmentation(image: np.ndarray, BINARY=True, Normalize=False) -> np.ndarray:
    state = torch.load('./model/model.pth', map_location='cpu')
    model = U_Net(1, 1)
    model.load_state_dict(state['net'])
    data = torch.Tensor(image/255.).unsqueeze(0).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(data)
    prediction = prediction.data.numpy().squeeze()
    if BINARY:
        if Normalize:
            prediction[prediction >= 0.5] = 1
            prediction[prediction < 0.5] = 0
        else:
            prediction[prediction >= 0.5] = 255
            prediction[prediction < 0.5] = 0
    return prediction


def FindContour(image: np.ndarray) -> np.ndarray:
    '''
    利用OpenCV找出二值图的轮廓
    输入图像的数据类型需为8U1C
    输出为轮廓的坐标
    '''
    contours, _ = cv2.findContours(image.astype(
        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0].squeeze()


def Visualization(image: np.ndarray, int_contours: np.ndarray, ext_contours: np.ndarray = None):
    plt.imshow(image, cmap='gray')
    plt.plot(int_contours[:, 0], int_contours[:, 1], '-', c='g')
    # plt.plot(ext_contours[:,0],ext_contours[:,1],'-',c='y')
    plt.show()


def main():
    dataset = ReadData()
    sample = dataset.image_data[0]
    GT = dataset.mask_data[0]
    int_predict = InternalSegmentation(sample)
    contours = FindContour(int_predict)
    Visualization(sample, contours)


if __name__ == "__main__":
    main()
