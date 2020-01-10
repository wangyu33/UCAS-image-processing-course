import matplotlib.pyplot as plt
import numpy as np

def ostu(gray_fig):
    """
    该函数通过“大津法”求得适合图像二值化的最佳阈值并返回
    :param gray_fig:待进行二值化处理的灰度图像
    :return: 通过“大津法”求得的最佳阈值
    """
    thres_list=np.zeros(256,dtype=float)
    #利用循环，对所有灰度逐次尝试以寻找最佳灰度
    for threshold in range(256):
        bin_fig1=gray_fig>threshold          #灰度值大于阈值的像素置1
        bin_fig0=gray_fig<=threshold         #灰度值小于阈值的像素置1
        foreground_num=np.sum(bin_fig1)      #灰度图像中大于阈值的元素个数
        background_num=np.sum(bin_fig0)      #灰度图像中小于阈值的元素个数
        if 0==foreground_num:   #若从某阈值开始前景图像像素个数为0，则停止寻找
            break
        if 0==background_num:   #若从某阈值下背景图像像素个数为0，则直接进行下一次尝试
            continue
        w0=foreground_num/gray_fig.size
        u0=np.sum(gray_fig * bin_fig1)/foreground_num
        w1=background_num/gray_fig.size
        u1=np.sum(gray_fig * bin_fig0) / background_num
        thres_list[threshold]=w0*w1*(u0 - u1)*(u0 - u1)
    thres=np.argmax(thres_list)      #找出最合适的阈值
    return thres

def to_bin(fig,threshold):
    """
    将图片进行二值化处理
    :param fig: 待二值化的图片
    :param threshold:进行二值化的阈值
    :return: 二值化处理好后的图像
    """
    m,n=fig.shape
    fig[fig>threshold] = 1
    fig[fig<threshold] = 0
    plt.imshow(fig,cmap="gray")
    plt.show()
    return fig


def Pad_replicate(image,f):
    pad_size = int((f-1)/2)
    h = image.shape[0] #图像的高
    w = image.shape[1] #图像的宽

    w_pad_u = image[0,:].copy().reshape(1,w) #填充在原图像的上、下 注意这里对使用切片的部分进行copy
    pad_image = image.copy()
    for _ in range(pad_size):
        pad_image = np.concatenate((w_pad_u,pad_image),axis=0)

    w_pad_d = image[h-1,:].copy().reshape(1,w) #填充在原图像的上、下 注意这里对使用切片的部分进行co
    for _ in range(pad_size):
        pad_image = np.concatenate((pad_image,w_pad_d),axis=0)

    h_pad_l = image[:,0].copy().reshape(h,1) #填充在原图像的左、右 先填充上下的话 左右填充时需要多填
    for _ in range(pad_size): #左上角元素填充
        h_pad_l = np.concatenate((image[0,0].reshape(1,1),h_pad_l),axis=0)
    for _ in range(pad_size): #左下角元素填充
        h_pad_l = np.concatenate((h_pad_l,image[h-1,0].reshape(1,1)),axis=0)

    h_pad_r = image[:, w - 1].copy().reshape(h, 1)  # 填充在原图像的左、右 先填充上下的话 左右填充时需要多
    for _ in range(pad_size): #右上角元素填充
        h_pad_r = np.concatenate((image[0, w - 1].reshape(1, 1), h_pad_r), axis=0)
    for _ in range(pad_size):  # 右下角元素填充
        h_pad_r = np.concatenate((h_pad_r, image[h - 1, w - 1].reshape(1, 1)), axis=0)

    for _ in range(pad_size): #右下角元素填充
        pad_image = np.concatenate((h_pad_l,pad_image),axis=1)
        pad_image = np.concatenate((pad_image,h_pad_r),axis=1)
    return pad_image


def Crosion(image,cov,pad="zeros"):
    f = cov.shape[0] #卷积核大小
    pad_size = int((f-1)/2)
    h = image.shape[0] #图像的高
    w = image.shape[1] #图像的宽
    if pad == "zeros":
        w_pad = np.zeros([pad_size,w])   #填充在原图像的上、下
        h_pad = np.zeros([h+2*pad_size,pad_size]) #填充在原图像的左、右 先填充上下的话 左右填充时需要
        temp = np.concatenate((w_pad,image),axis=0) #上填充
        temp = np.concatenate((temp,w_pad),axis=0) #下填充
        temp = np.concatenate((temp,h_pad),axis=1) #右填充
        temp = np.concatenate((h_pad,temp),axis=1) #左填充
    elif pad == "ones":
        temp = Pad_replicate(image,f)
    pad_image = temp.copy()
    h = pad_image.shape[0] #图像的高
    w = pad_image.shape[1] #图像的宽
    hh = (h-f)+1 #卷积后图像高
    ww = (w-f)+1 #卷积后图像宽
    cov_image = np.zeros([hh,ww])
    for i in range(0,h-f+1): #对原图像高进行遍历
        for j in range(0,w-f+1): #对原图像宽进行遍历
            cov_image[i,j] = int(np.sum(pad_image[i:i+f,j:j+f]*cov)==np.sum(np.sum(cov))) #计算卷积
    return cov_image

def HMT(A,b1,b2):
    """
    b1腐蚀A ∩ b2腐蚀A补
    """
    height,weight = A.shape
    A_cro = Crosion(A,b1) #b1腐蚀A
    A_c = np.ones((height,weight))-A #A的补
    A_c_cro = Crosion(A_c,b2) #b2腐蚀A的补
    return A_cro*(A_cro==A_c_cro) #返回交集


def pad(f, w,p):
    '''
    该函数用于实现二维卷积运算
    param f: 待进行卷积操作的图片
    param w: 卷积核
    return: 填充后的图片
    '''
    fig = f
    a, b = fig.shape   #a表示图片的高度，b表示图片的宽度
    m, n = w.shape     #m表示卷积核的高度，n表示卷积核的宽度
    #创造一个与边缘填充后图片大小相同的零矩阵，作为填充模板
    padding_fig_0 = np.zeros((a+m-1,b+n-1),dtype=int)
    padding_fig_1 = np.ones((a + m - 1, b + n - 1), dtype=int)
    #因为0填充或1填充对边界腐蚀、膨胀效果有差异，所以会根据特定情况进行选择
    if p == 'zeros':  # 零填充方式
        padding_fig_0[(m - 1) // 2:(m - 1) // 2 + a, (n - 1) // 2:(n - 1) // 2 + b] = fig
        return padding_fig_0
    elif p == 'ones':  # 最近边界灰度值填充方式
        padding_fig_1[(m - 1) // 2:(m - 1) // 2 + a, (n - 1) // 2:(n - 1) // 2 + b] = fig
        return padding_fig_1


def corrode(A,B,p):
    a,b=A.shape
    m,n=B.shape
    B_sum=np.sum(B)
    A_pad=pad(A,B,p)
    new_fig=np.zeros((a+m-1,b+n-1),dtype=int)
    for i in range((m-1)//2,(m-1)//2+a):
        for j in range((n-1)//2,(n-1)//2+b):
            if B_sum==np.sum(B*A_pad[i-(m-1)//2:i+(m+1)//2,j-(n-1)//2:j+(n+1)//2]):
                new_fig[i, j] = 1
    # 从new_fig中截取出与原图对应的部分，去除填充部分
    new_fig = new_fig[(m-1)//2:(m-1)//2+a,(n-1)//2:(n-1)//2+b]
    return new_fig




def hitting(A,B1,B2):
    a,b=A.shape
    Ac=np.ones((a,b))-A
    A_D=corrode(A,B1,"zeros")
    # plt.imshow(A_D, cmap='gray')
    # plt.show()
    Ac_WD=corrode(Ac,B2,"ones")
    # plt.imshow(Ac_WD, cmap='gray')
    # plt.show()
    A_hit_B=(A_D==Ac_WD)*A_D   #A_D和Ac_WD的交集作为击中或击不中操作的结果
    # plt.imshow(A_hit_B, cmap="gray")
    # plt.show()
    return A_hit_B

def fig_thin(A):
    B1_D=np.array([[0,0,0],[0,1,0],[1,1,1]])
    B1_WD = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    B2_D=np.array([[0,0,0],[1,1,0],[1,1,0]])
    B2_WD = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    B3_D= np.rot90(B1_D,k=-1)
    B3_WD = np.rot90(B1_WD, k=-1)
    B4_D= np.rot90(B2_D, k=-1)
    B4_WD = np.rot90(B2_WD, k=-1)
    B5_D= np.rot90(B1_D, k=-2)
    B5_WD = np.rot90(B1_WD, k=-2)
    B6_D= np.rot90(B2_D, k=-2)
    B6_WD = np.rot90(B2_WD, k=-2)
    B7_D= np.rot90(B1_D, k=-3)
    B7_WD = np.rot90(B1_WD, k=-3)
    B8_D= np.rot90(B2_D, k=-3)
    B8_WD = np.rot90(B2_WD, k=-3)
    B=[(B1_D,B1_WD),(B2_D,B2_WD),(B3_D,B3_WD),(B4_D,B4_WD),(B5_D,B5_WD),(B6_D,B6_WD),(B7_D,B7_WD),(B8_D,B8_WD)]
    new_A = A.copy()
    while True:
        for B_D,B_WD in B:
            new_A=new_A-hitting(new_A,B_D,B_WD)
        plt.imshow(new_A, cmap='gray')
        plt.show()
        if (new_A==A).all():
            break
        else:
            A=new_A.copy()
    return new_A

def main():
    pass


if __name__=="__main__":
    fig=plt.imread('bone3.png')
    A=fig[:,:,0]
    A=to_bin(A,0.1)
    # A=np.array([[1,1,1,1,1,1,1,1,1,1,1],
    #             [1,1,1,1,1,1,1,1,1,0,0],
    #             [1,1,1,1,1,1,1,1,1,0,0],
    #             [1,1,1,1,1,1,1,1,1,0,0],
    #             [1,1,1,0,0,1,1,1,1,0,0]])
    A[0:5,25:35]=0
    A=fig_thin(A)
    plt.imshow(A,cmap='gray')
    plt.show()
    A = np.array([[0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],[1, 0, 0, 0, 1, 1, 1],[0, 0, 0, 0, 0, 0, 0]])
    B = np.array([[1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1]])

