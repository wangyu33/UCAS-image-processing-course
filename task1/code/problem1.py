import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def scanLine4e(f, I, loc):
    #f:gray level image
    #I -> int    loc -> str
    if loc == 'row':
        return f[I,:]

    if loc == 'column':
        return f[:,I]

    return '参数错误'




def run(filename):
    f = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    r = np.size(f,0)
    c = np.size(f,1)
    plt.figure(filename)
    plt.imshow(f, cmap = 'gray')
    plt.show()
    f_r = r//2          #求中心行的index，当行数为偶数时取较大中心行
    f_c = c//2          #求中心列的index，当列数为偶数时取较大中心列
    r_vector = scanLine4e(f,f_r,'row')
    c_vector = scanLine4e(f,f_c,'column')
    xr = np.linspace(1,r,r)
    xc = np.linspace(1,c,c)
    plt.figure()
    plt.plot(xc, r_vector, 'bo--', ms = 2, label = 'row pixel')
    plt.plot(xr, c_vector, 'ro--', ms = 2, label='column pixel')
    plt.legend(loc = 1)
    plt.ylabel('pixel value')
    plt.title(filename)
    plt.show()

def main():
    run('cameraman.tif')
    run('einstein.tif')

if __name__ == '__main__':
    main()