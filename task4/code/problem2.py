import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

#添加椒盐噪声,SNR为信噪比
def sp_noise(pic, SNR = 0.95):
    f = pic.copy()
    mask = np.random.choice((0, 1, 2), size = f.shape, p = [SNR, (1 - SNR) / 2, (1 - SNR) / 2])
    f[mask == 1] = 255
    f[mask == 2] = 0
    return f

def myfilter(temp):
    #有选择保边缘平滑法滤波器
    #生成三个不同掩模
    mod1 = np.zeros(temp.shape, dtype = float)
    mod1[1:4, 1:4] = 1
    mod2 = np.zeros(temp.shape, dtype = float)
    mod2[0:2, 1:4] = 1
    mod2[2,2] = 1
    mod3 = np.zeros(temp.shape, dtype = float)
    mod3[0:2, 0:2] = 1
    mod3[1:3, 1:3] = 1

    temp1 = mod1 * (temp+0.0000000001)
    temp1 = temp1.flatten()
    temp1 = temp1[np.nonzero(temp1)]
    var = np.var(temp1)
    mean = np.mean(temp1)

    #通过旋转得到另外八个掩模的方差
    for i in range(4):
        temp1 = mod2 * (temp + 0.0000000001)
        temp1 = temp1.flatten()
        temp1 = temp1[np.nonzero(temp1)]
        var1 = np.var(temp1)
        if var1 < var:
            mean = np.mean(temp1)
        mod2 = np.rot90(mod2, 1)
        temp1 = mod3 * (temp + 0.0000000001)
        temp1 = temp1.flatten()
        temp1 = temp1[np.nonzero(temp1)]
        var1 = np.var(temp1)
        if var1 < var:
            mean = np.mean(temp1)
        mod3 = np.rot90(mod3, 1)
    return int(mean)

def choice_smooth(pic):
    row, col = np.shape(pic)
    f = pic.copy()
    #图像边缘填充，以镜像填充
    f_pad = cv2.copyMakeBorder(f, 2, 2, 2, 2, cv2.BORDER_DEFAULT)
    ans = np.zeros(pic.shape, dtype = 'uint8')
    for i in range(row):
        for j in range(col):
            temp = f_pad[i:i+5, j:j+5]
            ans[i][j] = myfilter(temp)
            print('procesing the %d %d pixel' % (i,j))
    return ans

def main(filename):
    pic = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    f = sp_noise(pic)
    # f_ = choice_smooth(f)
    plt.subplot(1, 3, 1)
    plt.imshow(pic, cmap = 'gray')
    plt.title('original picture')
    plt.subplot(1, 3, 2)
    plt.imshow(f, cmap='gray')
    plt.title('picture after adding sp noise')
    # plt.subplot(1, 3, 3)
    # plt.imshow(f_, cmap='gray')
    # plt.title('picture after choice smooth')
    plt.show()

if __name__ == '__main__':
    main('lena.tiff')