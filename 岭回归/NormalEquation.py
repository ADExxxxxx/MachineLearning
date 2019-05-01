import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("longley.csv", delimiter=',')
x_data = data[1:, 2:]
y_data = data[1:, 1, np.newaxis]

# 此处可知样本数量为16， 特征数量为6
print("x_data的维度", x_data.shape)
print("y_data的维度", y_data.shape)
# 添加偏置单元
X = np.concatenate((np.ones((16, 1)), x_data), axis=1)
# 类似于添加一个全为1的x0,为了计算出θ0的值
print("添加偏置单元后的维度", X .shape)


# 计算权重向量
def weights(xArr, yArr, lamda=0.2):
    print("xArr的类型:", type(xArr))
    xMat = np.mat(xArr)     # 将一个数组转化成一个矩阵，这样方便在之后可以直接使用矩阵乘法
    print("xMat的类型", type(xMat))
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat    # 相当于 X.T * X
    print("xTx的维度:", xTx.shape)
    # 正规化项
    rxTx = xTx + np.eye(xMat.shape[1]) * lamda  # 正则化项: (X.T * X + λ * I)
    print("rxTx的维度为:", rxTx.shape)
    # 计算矩阵的值，如果值为0，则说明该矩阵没有逆矩阵
    if np.linalg.det(rxTx) == 0.0:
        print("This martix can't do inverse")
        return
    # 参数向量求解
    # Weight = (X.T * X + λ * I)^(-1) * X.T * y
    # X:输入的原始数据矩阵(m*n),m为样本个数，
    W = rxTx.I * xMat.T * yMat
    return W


# 此处W相当于模型训练好的系数
W = weights(X, y_data)
print("Weights的维度:", W.shape)
print("X的维度:", X.shape)
# 计算预测值
print("预测值:", np.mat(X) * np.mat(W))
print("真实值:", y_data)
print("误差值:", np.mat(X) * np.mat(W) - y_data)
print("误差率:", abs((np.mat(X) * np.mat(W) - y_data) / y_data * 100))