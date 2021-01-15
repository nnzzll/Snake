import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from test import Filter, getContour


sample = cv2.imread('./IM000001_0005.png', 0)
img_filter = Filter(copy.deepcopy(sample).astype(float))
gaussian = img_filter.Gaussian()
init_contour = getContour(gaussian)
x, y = init_contour[:, 0], init_contour[:, 1]
print(x.shape)
print(y.shape)
max_iteration = 2000
alpha = 0.2
beta = 0.2
gamma = 1
kappa = 0.1
wl = 0
we = 0.4
wt = 0
image = gaussian
row, col = image.shape
# 图像力:线函数
Eline = image
# 图像力:边函数
gx, gy = np.gradient(image)
Eedge = -1*np.sqrt(gx*gx+gy*gy)
# 图像力:终点函数
'''
'''
# 外部力: Eext = Eimage + Econ
Eext = wl*Eline + we*Eedge + wt
fx, fy = np.gradient(Eext)
m = x.shape[0]

# 计算五对角矩阵
b1 = beta
b2 = -(alpha+4*beta)
b3 = 2*alpha+6*beta
b4 = b2
b5 = b1

A = b1*np.roll(np.eye(m), 2, axis=0)
A = A + b2*np.roll(np.eye(m), 1, axis=0)
A = A + b3*np.roll(np.eye(m), 0, axis=0)
A = A + b4*np.roll(np.eye(m), -1, axis=0)
A = A + b2*np.roll(np.eye(m), -2, axis=0)
Ainv = np.linalg.inv(A+gamma*np.eye(m))


def interp(matrix, xs, ys):
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    from scipy.interpolate import interp2d
    f = interp2d(x, y, matrix)
    return f(xs, ys)


plt.imshow(sample,cmap='gray')
for i in range(max_iteration):
    ssx = gamma*x-kappa*(interp(fx,x,y).diagonal())
    ssy = gamma*y-kappa*(interp(fy,x,y).diagonal())
    x = Ainv @ ssx
    y = Ainv @ ssy
    plt.plot(x,y,'-')
    break
plt.show()
print("")