import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

#归并写FFT
def dft2D(f):
    #二维FFT   problem_1
    row, column = f.shape
    F = np.zeros([row, column], dtype = complex)
    for i in range(row):
        F[i, :] = np.fft.fft(f[i, :])
    for j in range(column):
        F[:, j] = np.fft.fft(F[:,j])
    return  F

def idft2D(F):
    #二维FFT逆变换 problem_2
    row, column = F.shape
    F = F.conjugate()
    f = dft2D(F)
    f = f / (row * column)
    f = f.conjugate()
    return f


def run(filename):
    # problem_3
    pic= cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    f = np.array(pic, dtype = 'float') / 255
    F = dft2D(f)
    g_ = idft2D(F)
    g_ = np.array(np.abs(g_), dtype = 'float')
    g = np.array(np.abs(g_) * 255, dtype = 'uint8')
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(pic, cmap ='gray')
    plt.title('original')
    plt.subplot(1, 3, 2)
    plt.imshow(g, cmap='gray')
    plt.title('FFT')
    plt.subplot(1, 3, 3)
    f_g = cv2.absdiff(f, g_)
    cv2.imwrite('f-g.tiff',cv2.absdiff(pic, g)) #保存f-g,还原到0-255
    plt.imshow(f_g, cmap='gray', vmin = 0, vmax = 1) #由于f为归一化后的图，所以imshow的值域为0-1
    plt.title('f - g')
    plt.show()

def creat_rectangle_pic():
    f = np.zeros([512,512], dtype = 'uint8')
    #f[26:186,:][:,151:361] = 255
    f[:,:]=0
    m,n=f.shape
    for i in range(m):
        for j in range(n):
            if (-1)**(i+j)==1:
                f[i,j]=1
    g=f.copy()
    cv2.imwrite('rectangle.tiff', f)
    # plt.figure()
    # plt.imshow(f, cmap='gray')
    # plt.show()


def main():
    #run('house.tif')
    run('rose512.tif')
    creat_rectangle_pic()

    f = cv2.imread('rectangle.tiff', cv2.IMREAD_GRAYSCALE)
    f = np.array(f, dtype='float') / 255
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(f, cmap='gray') #由于f为归一化后的图，所以imshow的值域为0-1
    plt.title('original image')
    plt.subplot(2,2,2)
    temp =np.abs(dft2D(f))
    plt.imshow(np.abs(dft2D(f)), cmap='gray')
    plt.title('FFT image')


    #中心化处理
    f = dft2D(f)
    row, column = f.shape
    f_center1 = f[row // 2:, :][:, column // 2:]
    f_center2 = f[row // 2:, :][:, :column // 2]
    f_center3 = f[:row // 2, :][:, column // 2:]
    f_center4 = f[:row // 2, :][:, :column // 2]
    f_center = np.vstack((np.hstack((f_center1, f_center2)), np.hstack((f_center3, f_center4))))
    # f_center = np.fft.fftshift(f)
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(f_center), cmap='gray')
    plt.title('center image')
    S = np.log(1 + np.abs(f_center))
    plt.subplot(2, 2, 4)
    plt.imshow(S, cmap='gray')
    plt.title('Spectral image')
    plt.show()


if __name__ == '__main__':
    main()

