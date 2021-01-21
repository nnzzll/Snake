import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, CubicSpline


def interp(matrix, xs, ys):
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    f = interp2d(x, y, matrix)
    result = f(xs, ys).diagonal()
    return result


def getContours(image: np.ndarray, max_points: int = 50, cmap: str = 'gray') -> np.ndarray:
    '''
    手动标记轮廓
    输入:图像数组,初始点的最大数量,图像的Color Map
    输出:经过插值后的轮廓坐标数组
    '''
    plt.imshow(image, cmap=cmap)
    position = plt.ginput(n=max_points)
    points_num = len(position) + 1
    position.append(position[0])
    position_ori = np.array(position, dtype=float)

    # 对手动标记的点进行3次样条插值
    t_ori = np.arange(1, points_num + 1)
    Spline = CubicSpline(t_ori, position, bc_type='periodic')
    t_new = np.arange(1, points_num+1, 0.1)
    position_new = Spline(t_new)
    position_new = np.array(position_new, dtype=float)

    # 显示标记结果
    plt.imshow(image, cmap='gray')
    plt.plot(position_ori[:, 0], position_ori[:, 1], 'o')
    plt.plot(position_new[:, 0], position_new[:, 1], '-')
    plt.show()

    return position_new


def snake(image: np.ndarray, contour: np.ndarray, alpha: float = 0.2, beta: float = 0.2, gamma: float = 1, kappa: float = 0.1, w_line: float = 0, w_edge: float = 0.4, max_iteration: int = 500):
    row, col = image.shape
    x, y = contour[:, 0], contour[:, 1]
    # 计算图像力的线函数与边函数
    Eline = image
    gx, gy = np.gradient(image)
    Eedge = -1*np.sqrt(gx*gx+gy*gy)

    # 外部力
    Eext = w_line*Eline + w_edge*Eedge

    # 计算五对角矩阵
    b1 = beta
    b2 = -(alpha+4*beta)
    b3 = 2*alpha+6*beta
    b4 = b2
    b5 = b1

    m = x.shape[0]
    A = b1*np.roll(np.eye(m), 2, axis=0)
    A = A + b2*np.roll(np.eye(m), 1, axis=0)
    A = A + b3*np.roll(np.eye(m), 0, axis=0)
    A = A + b4*np.roll(np.eye(m), -1, axis=0)
    A = A + b5*np.roll(np.eye(m), -2, axis=0)
    Ainv = np.linalg.inv(A+gamma*np.eye(m))

    # 迭代
    fx, fy = np.gradient(Eext)
    for i in range(max_iteration):
        ssx = gamma*x-kappa*(interp(fx, x, y))
        ssy = gamma*y-kappa*(interp(fy, x, y))
        x = Ainv @ ssx
        y = Ainv @ ssy
    return x, y
