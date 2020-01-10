#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : problem.py
# Author: WangYu
# Date  : 2019/12/27

import cv2
from matplotlib import pyplot as plt
import numpy as np
import math

eps = 1e-3

def blur_kernel(size, theta, offset):
# size 模糊核大小 theta 角度 offset 偏移量
    PSF = np.zeros(size)
    center = [int(size[0] / 2), int(size[1] / 2)]

    theta_cos = math.cos(theta * math.pi /180)
    theta_sin = math.sin(theta * math.pi /180)
    # print(theta_cos)
    # print(theta_sin)
    theta_sin = math.sin(theta * math.pi /180)
    for i in range(offset):
        PSF[int(center[0] - i * theta_sin), int(center[1] + i * theta_cos)] = 1
        PSF[int(center[0] + i * theta_sin), int(center[1] - i * theta_cos)] = 1
    return PSF

def inverse_filter(pic, PSF):
    #逆滤波
    pic_FFT = np.fft.fft2(pic)
    PSF_FFT = np.fft.fft2(PSF, s = pic_FFT.shape) + eps
    result = np.fft.ifft2(pic_FFT / PSF_FFT)
    result = np.abs(np.fft.fftshift(result))
    return result

def wiener_filter(pic, PSF, k = 0.1):
    #逆滤波
    pic_FFT = np.fft.fft2(pic)
    PSF_FFT = np.fft.fft2(PSF, s = pic_FFT.shape) + eps
    PSF_w = np.conj(PSF_FFT) / (np.abs(PSF_FFT)**2 + k)
    result = np.fft.ifft2(pic_FFT * PSF_w)
    result = np.abs(np.fft.fftshift(result))
    return result

def run(filename):
    #生成灰度图，也可以分别求三个通道
    f = cv2.imread(filename, cv2.IMREAD_COLOR)
    f = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

    # #求倒谱
    #1.求频谱图,并归一化
    f_fft = np.fft.fft2(f)
    #2.取log，压缩频谱
    f_fft = np.log(1 + np.abs(f_fft))
    #3.实现倒谱,三次方是为了使高频更为明显
    f_fft = np.abs(np.fft.ifft2(f_fft ** 3))
    H = np.log(1 + np.abs(f_fft))
    H = np.fft.fftshift(H)
    plt.imshow(H, cmap='gray')
    plt.title('cepstrum')
    plt.show()


    PSF1 = blur_kernel(f.shape, 120, 20.5)
    PSF2 = blur_kernel(f.shape, 0, 0)
    PSF3 = blur_kernel(f.shape, 90, 0)
    PSF = PSF1 + PSF2 + PSF3
    PSF = PSF / np.sum(PSF)

    # PSF = H
    # PSF = np.where(PSF < 0.1 * np.max(PSF), 0, PSF)
    # PSF = PSF / np.sum(PSF)

    #显示PSF
    plt.figure()
    plt.imshow(PSF, cmap='gray')
    plt.title('my PSF')
    plt.show()

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(f, cmap='gray')
    plt.title('original picture')
    plt.show()
    plt.subplot(1,3,2)
    #F1 = inverse_filter(f, PSF)
    #添加边界以消除振铃
    f = cv2.copyMakeBorder(f,128,128,128,128, cv2.BORDER_DEFAULT)
    F1 = wiener_filter(f, PSF, k = 0.01)
    temp = np.abs(np.fft.fftshift(np.fft.fft2(F1)))
    plt.imshow(np.log(1 + temp), cmap='gray')
    plt.title('the fft of image after wiener_filter')
    plt.show()


    plt.subplot(1,3,3)
    #图像截取原图部分
    F1 = F1[F1.shape[0] - 513:-1, F1.shape[1] - 513:-1]
    # kernel1 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32)/9
    # kernel2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    # F1 = cv2.filter2D(F1, -1, kernel=kernel1)
    # F1 = cv2.filter2D(F1, -1, kernel=kernel2)
    plt.imshow(F1, cmap='gray', vmax = 255, vmin = 0)
    plt.title('image after wiener_filter')
    plt.show()





if __name__ == '__main__':
    filename = 'test.png'
    run(filename)