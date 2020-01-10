# -*- coding: utf-8 -*-
import cv2
import math
import matplotlib.pyplot as plt


def run(filename):
    f = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('original picture')
    plt.imshow(f, cmap='gray')
    plt.show()
    sig = 1
    m = math.ceil(sig * 3) * 2 + 1
    f_cov = cv2.GaussianBlur(f, (m,m), 1)
    # cv2.imwrite('lena_sig1.tiff', f_cov)
    plt.subplot(1, 2, 2)
    plt.title('sig = %d' % (sig))
    plt.imshow(f_cov, cmap='gray')
    plt.show()

def main():
    run('cameraman.tif')
    run('einstein.tif')
    run('lena.tiff')
    run('mandril.tif')
    f1 = cv2.imread('lena_mysig1.tiff', cv2.IMREAD_GRAYSCALE)
    f2 = cv2.imread('lena_sig1.tiff', cv2.IMREAD_GRAYSCALE)
    ff = cv2.absdiff(f1,f2)
    plt.figure()
    plt.subplot(1,3,1)
    plt.title('my conv')
    plt.imshow(f1, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('cv2 conv')
    plt.imshow(f2, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('the D-value of lena')
    plt.imshow(cv2.absdiff(f2, f1), cmap='gray')

    f3 = cv2.imread('lena_zero.tiff', cv2.IMREAD_GRAYSCALE)
    f4 = cv2.imread('lena_replicate.tiff', cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('zero method')
    plt.imshow(f3, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('replicate method')
    plt.imshow(f4, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('the D-value of two method')
    plt.imshow(255-cv2.absdiff(f4, f3), cmap='gray')

    f5 = cv2.imread('cameraman_zero.tiff', cv2.IMREAD_GRAYSCALE)
    f6 = cv2.imread('cameraman_replicate.tiff', cv2.IMREAD_GRAYSCALE)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.title('zero method')
    plt.imshow(f5, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('replicate method')
    plt.imshow(f6, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('the D-value of two method')
    plt.imshow(cv2.absdiff(f6,f5), cmap='gray')



if __name__ == '__main__':
    main()