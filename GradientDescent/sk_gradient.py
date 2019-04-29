from sklearn.linear_model import  LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 数据导入
data = np.genfromtxt("data.csv", delimiter=",")
# print(data)

# 数据分割
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]

# 绘制散点图
plt.scatter(x_data, y_data)


# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)

plt.plot(x_data, model.predict(x_data), 'r')
plt.show()