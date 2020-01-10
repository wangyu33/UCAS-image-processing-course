import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def laplacian_enhance():
    #拉普拉斯增强算子
    L_filter = np.zeros((3,3), dtype = float)
    L_filter[1, 1] = 5
    L_filter[(0,2), 1] = -1
    L_filter[1, (0,2)] = -1
    return L_filter

def cov(pic):
    row, col = np.shape(pic)
    f = pic.copy()
    L_filter = laplacian_enhance()
    #滤波器对称
    # 图像边缘填充，以镜像填充
    f = cv2.copyMakeBorder(f, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    f_pad = np.array(f, dtype = float)
    ans = np.zeros(pic.shape, dtype = float)
    for i in range(row):
        for j in range(col):
            temp = f_pad[i:i+3, j:j+3]
            temp = np.sum(temp * L_filter)
            if temp < 0:
                ans[i][j] = 0
            elif temp >255:
                ans[i][j] = 255
            else:
                ans[i][j] = temp
            print('procesing the %d %d pixel' % (i,j))

    return np.array(ans, dtype = 'uint8')


def main(filename):
    pic = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    f = cov(pic)
    plt.subplot(1, 2, 1)
    plt.imshow(pic, cmap='gray')
    plt.title('original picture')
    plt.subplot(1, 2, 2)
    plt.imshow(f, cmap='gray')
    plt.title('picture after laplacian_enhance')
    plt.show()

if __name__ == '__main__':
    #main('lena.tiff')
    main('cameraman.tif')