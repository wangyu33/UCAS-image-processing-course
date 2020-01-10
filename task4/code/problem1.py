import numpy as np
import cv2
import matplotlib.pyplot as plt

#该程序为直方图均衡化处理,可用于增加图像对比度
def myhisteq(pic):
    row, col = np.shape(pic)
    imhist, bins = np.histogram(pic.flatten(), 256)
    cdf = np.zeros(imhist.shape, dtype = int)
    cdf[0] = imhist[0]
    #求直方图频数积分
    for i in range(1,256):
        cdf[i] = cdf[i - 1] + imhist[i]
    cdf = 255 * cdf / cdf[255]
    #向下取整
    cdf = np.array(cdf, dtype = 'uint8')
    ans = np.zeros(pic.shape, dtype = 'uint8')
    #映射
    for i in range(row):
        for j in range(col):
            ans[i][j] = cdf[pic[i][j]]
    return ans

def main(filename):
    f = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    f1 = myhisteq(f)
    # f2 = cv2.equalizeHist(f)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(f, cmap='gray')
    plt.title('original picture')
    plt.subplot(2, 2, 2)
    plt.hist(f.flatten(), bins=256, alpha=0.7)
    plt.title('hist of original picture')
    plt.subplot(2, 2, 3)
    plt.imshow(f1, cmap='gray')
    plt.title('picture after histeq')
    plt.subplot(2, 2, 4)
    plt.hist(f1.flatten(), bins=256, alpha=0.7)
    plt.title('hist of picture after histeq')
    plt.show()

if __name__ == '__main__':
    #main('lena.tiff')
    main('cameraman.tif')