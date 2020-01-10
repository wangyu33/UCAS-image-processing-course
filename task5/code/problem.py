import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from SE import *

def get_pic():
    #获取指纹灰度图,网上下载图指纹为黑背景为白，需要做反转
    f = cv2.imread('finger.jpg', cv2.IMREAD_COLOR)
    f_g = cv2.cvtColor(255 - f, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('finger.tiff', f_g)
    f = cv2.imread('bone3.png', cv2.IMREAD_COLOR)
    f_g = cv2.cvtColor(np.uint8(f > 128), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('bone.tiff', f_g)

def mythreshold(pic, x = 128):
    #图像二值化，像素大于阈值x时置于1，x默认为10
    f = pic > x
    f = np.array(f, dtype = 'uint')
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(pic, cmap = 'gray', vmin = 0, vmax = 255)
    plt.title('grayscale picture')
    plt.subplot(1, 2, 2)
    plt.imshow(f, cmap='gray', vmin = 0, vmax = 1)
    plt.title('binary picture')
    plt.show()
    return f

def myimdilate(f, SE, iter = 1):
    #SE为膨胀核 iter为膨胀次数
    SE = np.uint8(SE)
    f = np.uint8(f)
    sum = np.sum(SE)
    for i in range(iter):
        temp = cv2.filter2D(f, -1, SE)
        f = np.uint8(temp > 0)
    return np.uint8(f)

def myimerode(f, SE, iter = 1):
    #SE为膨胀核 iter为膨胀次数
    SE = np.uint8(SE)
    f = np.uint8(f)
    sum = np.sum(SE)
    for i in range(iter):
        temp = cv2.filter2D(f, -1, SE)
        f = np.uint8(temp >= sum)
    return np.uint8(f)

def myimopen(f, SE):
    f = myimerode(f, SE)
    f = myimdilate(f, SE)
    return np.uint8(f)

def myimclose(f, SE):
    f = myimdilate(f, SE)
    f = myimerode(f, SE)
    return np.uint8(f)

def mybwhitmiss(f, SE1, SE2):
    #击中击不中
    f_ = 1 - f
    f1 = myimerode(f, SE1)
    f2 = myimerode(f_, SE2)
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(f1, cmap='gray', vmin=0, vmax=1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(f2, cmap='gray', vmin=0, vmax=1)
    # plt.show()
    return np.uint8(f1 * f2)


def morphology_skeleton(f, SE = B):
    #x形态学骨骼提取
    while 1:
        flag = 0    #x循环终止符
        for i in range(8):
            B_1 = np.uint8(SE[i * 3: i * 3 + 3] > 0)
            B_2 = np.uint8(SE[i * 3: i * 3 + 3] < 0)
            temp = mybwhitmiss(f, B_1, B_2)
            if np.sum(temp) > 0:
                #迭代至无变化
                flag = 1
            f = f * (1 - temp)
        if flag == 0:
            break
    #f = remove_connect(f)
    return np.uint8(f)

def dst(edge, i, j):
    #使用杨戈老师讲到的距离变换方法
    ans = 10000
    for m in edge:
        ans = min(ans, (m[0]-i)*(m[0]-i)+(m[1]-j)*(m[1]-j))
    return ans

def dst_P(f, a = 3, b = 4):
    #使用彭思龙 老师课上的距离变换
    row, col = f.shape
    for i in range(1, row):
        for j in range(1, col -1):
            temp0 = f[i][j]
            temp1 = min(f[i][j - 1] + a, temp0)
            temp2 = min(f[i - 1][j - 1] + b, temp1)
            temp3 = min(f[i - 1][j] + a, temp2)
            temp4 = min(f[i - 1][j + 1] + b, temp3)
            f[i][j] = temp4

    for i in range(row - 1, -1, 1):
        for j in range(col - 1, -1, 2):
            temp0 = f[i][j]
            temp1 = min(f[i][j + 1] + a, temp0)
            temp2 = min(f[i + 1][j + 1] + b, temp1)
            temp3 = min(f[i + 1][j] + a, temp2)
            temp4 = min(f[i + 1][j - 1] + b, temp3)
            f[i][j] = temp4
    return f

def dst_skeleton(f, area = 8):
    #距离变换骨骼提取
    #取领域的极大值时领域所设置的范围
    edge = f - myimerode(f, SE1)#背景增强
    row, col = f.shape
    edge_ = []
    for m in range(row):
        for n in range(col):
            if edge[m][n] == 1:
                edge_.append([m,n])
    distance = np.array(f, dtype = float)
    #使用杨戈老师方法
    #============================================
    for i in range(row):
        for j in range(col):
            if f[i][j] == 1:
                distance[i][j] = dst(edge_, i, j)
                print('procesing the %d %d pixel' % (i,j))
    #============================================
    '''
    使用彭思龙老师方法
    #背景增强
    distance = edge + (f - edge) * 100
    distance = dst_p(distance)
    '''

    ans = np.zeros(distance.shape, dtype = 'uint8')
    r = area // 2
    for i in range(r, row - r):
        for j in range(r, col - r):
            temp = distance[i - r: i + r + 1, j - r: j + r + 1]
            if distance[i][j] == 0:
                continue
            index = np.where(temp == np.max(temp))
            ans[i - r + index[0], j - r + index[1]] = 1
    return np.uint8(ans > 0)

def cut_out(f, iter = 3, SE = C):
    # iter 为细化膨胀次数 SE为裁剪结构元组
    # 细化求X1
    X1 = f.copy()
    for k in range(iter):
        for i in range(8):
            B_1 = np.uint8(SE[i * 3:i * 3 + 3] > 0)
            B_2 = np.uint8(SE[i * 3:i * 3 + 3] < 0)
            temp = mybwhitmiss(X1, B_1, B_2)
            X1 = X1 * (1 - temp)
    B_1 = np.uint8(SE[0: 3] > 0)
    B_2 = np.uint8(SE[0: 3] < 0)
    # 求取端点X2
    X2 = mybwhitmiss(X1, B_1, B_2)
    for i in range(1, 8):
        B_1 = np.uint8(SE[i * 3:i * 3 + 3] > 0)
        B_2 = np.uint8(SE[i * 3:i * 3 + 3] < 0)
        temp = mybwhitmiss(X1, B_1, B_2)
        X2 = X2 + temp
        X2 = np.uint8(X2 > 0)
    # 通过迭代膨胀求X3
    X3 = X2
    for i in range(iter):
        X3 = myimdilate(X3, SE2) * f
    # 与骨架图求并集求X4
    X4 = X3 + X1
    return np.uint8(X4 > 0)


def main():
    #problem 1-2
    finger = cv2.imread('finger.tiff', cv2.IMREAD_GRAYSCALE)
    f = mythreshold(finger)

    #problem3
    # f_2 = morphology_skeleton(f)
    # f_3 = dst_skeleton(f)
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(f, cmap='gray', vmin=0, vmax=1)
    # plt.title('fingerprint picture')
    # plt.subplot(1, 3, 2)
    # plt.imshow(f_2, cmap='gray')
    # plt.title('fingerprint after morphology skeleton')
    # plt.subplot(1, 3, 3)
    # plt.imshow(f_3, cmap='gray')
    # plt.title('fingerprint after dst skeleton')
    # plt.show()

    bone = cv2.imread('bone.tiff', cv2.IMREAD_GRAYSCALE)
    # bone = np.zeros([32,42])
    # bone[1:31,1:41] = 1
    bone_2 = morphology_skeleton(bone)
    bone_3 = dst_skeleton(bone)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(bone, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Chinese character picture')
    plt.subplot(1, 3, 2)
    plt.imshow(bone_2, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Chinese character after morphology skeleton')
    plt.subplot(1, 3, 3)
    plt.imshow(bone_3, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Chinese character after dst skeleton')
    plt.show()

    #problem4
    # f_4 = cut_out(np.array(f_2, dtype = 'uint8'), 5)
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(f_2, cmap='gray')
    # plt.title('fingerprint skeleton picture')
    # plt.subplot(1, 3, 2)
    # plt.imshow(f_4, cmap='gray')
    # plt.title('fingerprint skeleton after cut out')
    # plt.subplot(1, 3, 3)
    # plt.imshow(cv2.absdiff(f_2,f_4), cmap='gray')
    # plt.title('absdiff of two picture')
    # plt.show()

    bone_4 = cut_out(np.array(bone_2, dtype = 'uint8'), 5)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(bone_2, cmap='gray', vmin = 0, vmax = 1)
    plt.title('Chinese character skeleton picture')
    plt.subplot(1, 3, 2)
    plt.imshow(bone_4, cmap='gray')
    plt.title('Chinese character skeleton after cut out')
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.absdiff(bone_2, bone_4), cmap='gray')
    plt.title('absdiff of two picture')
    plt.show()

if __name__ == '__main__':
    get_pic()
    main()