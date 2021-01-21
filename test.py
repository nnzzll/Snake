import cv2
from scipy.ndimage.filters import gaussian_filter
from snake import *


def main():
    sample = cv2.imread('./IM000001_0005.png', 0)
    sample = sample/255.
    gaussian = gaussian_filter(sample,sigma=1)
    init_contour = getContours(sample)
    begin = time.time()
    x, y = snake(gaussian, init_contour, alpha=0.2,
                 beta=0.2, w_edge=0.4, w_line=0, max_iteration=1000)
    print("计算用时:{:.3f}s.".format(time.time()-begin))
    plt.imshow(sample, cmap='gray')
    plt.plot(init_contour[:, 0], init_contour[:, 1], '--r', lw=1)
    plt.plot(y, x, '-b', lw=2)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    main()
