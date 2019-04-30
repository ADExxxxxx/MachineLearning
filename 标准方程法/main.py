import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt("data.csv", delimiter=',')
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]

# array的合并
# 1 按行合并
# 0 按列合并
X = np.concatenate((np.ones((100, 1)), x_data), axis=1)
# 经过合并生成[[x0, x1]]


def cal_weights(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    xTx = xMat.T * xMat
    # 计算矩阵的值值，如果值为0，说明没有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("This Matrix can't do inverse")
        return
    W = xTx.I * xMat.T * yMat
    return W


W = cal_weights(X, y_data)
# W[0]:biases , W[1]:Weights
print(W)
plt.scatter(x_data, y_data)
y_pre = x_data * W[1] +W[0]
plt.plot(x_data, y_pre, 'r')
plt.show()