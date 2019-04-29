import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读入数据
data = np.genfromtxt("data.csv", delimiter=',')
x_data = data[:, :-1]
y_data = data[:, -1]

lr = 0.0001     # 学习率
theta0 = 0
theta1 = 0
theta2 = 0
epochs = 1000   # 最大迭代次数


# 最小二乘法求代价函数
# J(θ) = (1/2*m) * (sum[i=1->m](h(xi)-y)^2
# h(θ) = θ2x2 + θ1x1 + θ0
def compute_cost(theta0, tehta1, theta2,  x_data, y_data):
    total_error = 0
    for i in range(0, len(x_data)):
        total_error += (y_data[i] - (theta1 * x_data[i, 0] + theta2 * x_data[i, 1] + theta0)) ** 2
    return total_error/(float(len(x_data)) * 2.0)


# 梯度下降过程
# θ := θ - α(dθ/dJ(θ))
# b := b - α(db/dJ(θ))
def gradient_descent_runner(x_data, y_data, theta0, theta1, theta2, lr, epochs):
    # 数据量
    m = float(len(x_data))
    for i in range(epochs):
        # 中间的梯度总和
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            theta0_grad += (1/m) * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0) - y_data[j])
            theta1_grad += (1/m) * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0) - y_data[j]) * x_data[j, 0]
            theta2_grad += (1/m) * ((theta1 * x_data[j, 0] + theta2 * x_data[j, 1] + theta0) - y_data[j]) * x_data[j, 1]
        # 更新参数
        theta0 = theta0 - (lr * theta0_grad)
        theta1 = theta1 - (lr * theta1_grad)
        theta2 = theta2 - (lr * theta2_grad)
    return theta0, theta1, theta2

print("Starting theta0 = {0}, theta1 = {1}, theta2 = {2}, error = {3}".format(theta0, theta1, theta2, compute_cost(theta0, theta1, theta2, x_data, y_data)))
print("Running....")
theta0, theta1, theta2 = gradient_descent_runner(x_data,y_data,theta0, theta1, theta2, lr, epochs)
print("After {0} iteration theta0 = {1}, theta1 = {2}, theta = {3}, error = {4}".format(epochs, theta0, theta1, theta2, compute_cost(theta0, theta1,theta2, x_data, y_data)))

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1],y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
print(x0)
x1 = x_data[:, 1]
print(x1)
# 生成网络矩阵
x0, x1 = np.meshgrid(x0, x1)
print((x0, x1))

z = theta0 + x0 * theta1 + x1 * theta2
# 画3D图
ax.plot_surface(x0, x1, z)
# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')

# 显示图像
plt.show()