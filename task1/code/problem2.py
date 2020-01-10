import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def run(filename):
    f = cv2.imread(filename, cv2.IMREAD_COLOR)
    plt.figure(filename)
    plt.subplot(1,2,1)
    rgb = rgb1gray(f,'average')
    plt.imshow(rgb, cmap='gray')
    plt.title('average method')
    plt.subplot(1, 2, 2)
    NTFC = rgb1gray(f)
    plt.imshow(NTFC, cmap='gray')
    plt.title('NTSC method')
    plt.show()

def main():
    run('lena512color.tiff')
    run('mandril_color.tif')

if __name__ == '__main__':
    main()