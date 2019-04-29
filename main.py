# 一元梯度下降样例
import numpy as np
import matplotlib.pyplot as plt

# 数据导入
data = np.genfromtxt("data.csv", delimiter=",")
# print(data)

# 数据分割
x_data = data[:, 0]
y_data = data[:, 1]

# 绘制散点图
# plt.scatter(x_data, y_data)

lr = 0.0001  # 定义学习率
b = 0        # 截距
k = 0        # 斜率
epochs = 50  # 最大的迭代次数

# 绘图
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(x_data, y_data)
plt.ion()


# 最小二乘法求代价函数
# J(θ) = (1/2*m) * (sum[i=1->m](h(xi)-y)^2
# h(θ) = kx + b
def compute_cost(b, k, x_data, y_data):
    total_error = 0
    for i in range(0, len(x_data)):
        total_error += (y_data[i] - k * x_data[i] + b) ** 2
    return total_error/(float(len(x_data)) * 2.0)


# 梯度下降过程
# θ := θ - α(dθ/dJ(θ))
# b := b - α(db/dJ(θ))
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    # 数据量
    m = float(len(x_data))
    for i in range(epochs):
        # 中间的梯度总和
        b_grad = 0
        k_grad = 0
        # 计算梯度的总和再求平均
        for j in range(0, len(x_data)):
            b_grad += (1/m) * ((k * x_data[j] + b) - y_data[j])
            k_grad += (1/m) * ((k * x_data[j] + b) - y_data[j]) * x_data[j]
        # 更新参数
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)
        if i % 2 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x_data, k * x_data + b, 'r')
            plt.pause(0.1)
            plt.show()

    return b, k


print("Starting b = {0}, k = {1}, error = {2}".format(b, k, compute_cost(b, k, x_data, y_data)))
print("Running......")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("After {0} iterations b = {1}, k = {2}, error = {3}".format(epochs, b, k, compute_cost(b, k, x_data, y_data)))

# 绘图
# plt.plot(x_data, y_data, 'b')
# plt.plot(x_data, k * x_data + b, 'r')
# plt.show()


