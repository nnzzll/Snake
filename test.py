from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from scipy.interpolate import CubicSpline
import ctypes
import copy
import os


class Filter:
    def __init__(self, inputs: np.ndarray, DLL_Path: str = os.getcwd()+"\\Lib.so"):
        self.Lib = ctypes.cdll.LoadLibrary(DLL_Path)
        self.inputs = inputs
        self.input_ptr = ctypes.cast(
            self.inputs.ctypes.data, ctypes.POINTER(ctypes.c_double))
        self.size = np.array(inputs.shape, dtype=int)
        self.size_ptr = ctypes.cast(
            self.size.ctypes.data, ctypes.POINTER(ctypes.c_int))

    def Median(self, kernel_size: int = 3) -> np.ndarray:
        outputs = np.zeros(self.inputs.shape)
        output_ptr = ctypes.cast(
            outputs.ctypes.data, ctypes.POINTER(ctypes.c_double))
        self.Lib.MedianFilter(self.input_ptr, output_ptr, self.size_ptr,
                              ctypes.c_int(kernel_size))
        return copy.deepcopy(outputs)

    def Average(self, kernel_size: int = 3) -> np.ndarray:
        outputs = np.zeros(self.inputs.shape)
        output_ptr = ctypes.cast(
            outputs.ctypes.data, ctypes.POINTER(ctypes.c_double))
        self.Lib.AverageFilter(self.input_ptr, output_ptr, self.size_ptr)
        return copy.deepcopy(outputs)

    def Gamma(self, gamma: float = 1.3) -> np.ndarray:
        outputs = np.zeros(self.inputs.shape)
        output_ptr = ctypes.cast(
            outputs.ctypes.data, ctypes.POINTER(ctypes.c_double))
        self.Lib.GammaCorrection(
            self.input_ptr, output_ptr, self.size_ptr, ctypes.c_double(gamma))
        return copy.deepcopy(outputs)

    def Gaussian(self, kernel_size: int = 3, sigma: float = 1) -> np.ndarray:
        outputs = np.zeros(self.inputs.shape)
        output_ptr = ctypes.cast(
            outputs.ctypes.data, ctypes.POINTER(ctypes.c_double))
        self.Lib.GaussianFilter(self.input_ptr, output_ptr, self.size_ptr, ctypes.c_int(
            kernel_size), ctypes.c_double(sigma))
        return copy.deepcopy(outputs)


def getContour(img: np.ndarray, max_points: int = 50) -> np.ndarray:
    '''
    手动标记初始轮廓
    输入:图像数组,初始点的最大数量
    输出:经过插值后的轮廓坐标数组
    '''
    plt.imshow(img, cmap='gray')
    # 手动标点，左键标记，右键结束标记
    position = plt.ginput(max_points)
    points_num = len(position) + 1
    position.append(position[0])
    position_ori = np.array(position, dtype=float)

    # 对手动标记的点进行3次样条插值
    t_ori = np.arange(1, points_num + 1)
    Spline = CubicSpline(t, position, bc_type='periodic')
    t_new = np.arange(1, points_num+1, 0.1)
    position_new = Spline(t_new)
    position_new = np.array(position_new, dtype=float)

    # 显示标记结果
    plt.imshow(img, cmap='gray')
    plt.plot(position_ori[:, 0], position_ori[:, 1], 'o')
    plt.plot(position_new[:, 0], position_new[:, 1], '-')
    plt.show()

    return position_new


def main():
    data = pydicom.read_file('./20190521/CASE134/RUN1/PDJAQ5VE')
    data_arr = data.pixel_array
    sample = data_arr[0, :, :]
    img_filter = Filter(copy.deepcopy(sample).astype(float))
    gaussian = img_filter.Gaussian()
    init_contour = getContour(gaussian)
    snake = active_contour(gaussian, init_contour, alpha=1,
                           beta=1, w_edge=0.4, w_line=0)
    plt.imshow(gaussian, cmap='gray')
    plt.plot(init_contour[:, 0], init_contour[:, 1], '--r', lw=1)
    plt.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    plt.axis("off")
    plt.show()
