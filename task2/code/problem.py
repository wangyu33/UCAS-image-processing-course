import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings

def rgb1gray(f,method = 'NTSC'):
    r = f[:, :, 0]
    g = f[:, :, 1]
    b = f[:, :, 2]
    if method == 'average':
        return (r//3 + g//3 + b//3)

    if method == 'NTSC':
        w = [0.2988,0.5870,0.1140]
        row = np.size(f, 0)
        column = np.size(f, 1)
        NTSC = np.zeros((row,column), dtype='uint8')
        for i in range(row):
            for j in range(column):
                NTSC[i,j] = w[0]*r[i,j] + w[1]*g[i,j] + w[2]*b[i,j]
        return NTSC
    return '参数错误'

def mypadding(pic, r, method = 'zero'):
    row, col = np.shape(pic)
    if method == 'replicate':
        matrix1 = np.tile([pic[0, 0]],(r,r))
        matrix2 = np.tile([pic[0, :]], (r, 1))
        matrix3 = np.tile([pic[0, col - 1]], (r, r))
        head = np.hstack((matrix1, matrix2, matrix3))
        matrix4 = np.tile(np.transpose([pic[:, 0]]), (1, r))
        matrix6 = np.tile(np.transpose([pic[:, col - 1]]), (1, r))
        body = np.hstack((matrix4, pic, matrix6))
        matrix7 = np.tile(pic[row - 1, 0], (r,r))
        matrix8 = np.tile([pic[row - 1, :]], (r, 1))
        matrix9 = np.tile(pic[row - 1, col - 1], (r,r))
        tail = np.hstack((matrix7, matrix8, matrix9))
        f = np.vstack((head, body, tail))
        return f

    elif method == 'zero':
        matrix1 = np.zeros((r,r), dtype = 'uint8')
        matrix2 = np.zeros((r,col), dtype = 'uint8')
        matrix3 = np.zeros((row, r), dtype='uint8')
        head = np.hstack((matrix1, matrix2, matrix1))
        body = np.hstack((matrix3, pic, matrix3))
        f = np.vstack((head, body, head))
        return f

    return '参数错误'

def gaussKernel(sig, m = 0):
    if m <= 0:
        m = math.ceil(sig * 3) * 2 + 1
    if m < 7:
        warnings.warn('m is too small', WarningType=UserWarning)
    kernel = np.zeros([m, m])
    sig = 2 * sig ** 2
    center = m // 2
    for i in range(m):
        for j in range(m):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / sig)
    return kernel / np.sum(kernel)

def twodConv(f, sig, w = 0):
    if w == 0:
        kernel = gaussKernel(sig)
    else:
        kernel = gaussKernel(sig, w)
    row, col = np.shape(f)
    f_padding = mypadding(f, kernel.shape[0] // 2, 'zero')
    for i in range(row):
        for j in range(col):
            temp = f_padding[i:i+kernel.shape[0],:][:, j:j+kernel.shape[0]]
            f[i][j] = np.sum(kernel*temp)
    return f

def run(filename):
    f = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.subplot(1,2,1)
    plt.title('original picture')
    plt.imshow(f, cmap = 'gray')
    plt.show()
    sig = 1
    f_cov = twodConv(f, sig)
    cv2.imwrite('lena_mysig1.tiff',f_cov)
    plt.subplot(1, 2, 2)
    plt.title('sig = %d' %(sig))
    plt.imshow(f_cov, cmap='gray')
    plt.show()


def main():
    # run('cameraman.tif')
    # run('einstein.tif')
    # f = cv2.imread('lena512color.tiff', cv2.IMREAD_COLOR)
    # f = rgb1gray(f)
    # cv2.imwrite('lena.tiff',f)
    # f = cv2.imread('mandril_color.tif', cv2.IMREAD_COLOR)
    # f = rgb1gray(f)
    # cv2.imwrite('mandril.tif',f)
    run('lena.tiff')
    # run('mandril.tif')



if __name__ == '__main__':
    main()



