import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# 数据处理
data = np.genfromtxt("data.csv", delimiter=',')

x_data = data[:, :-1]
y_data = data[:, -1]

# 建模计算
model = linear_model.LinearRegression()
model.fit(x_data, y_data)
# 系数
print("coefficients:", model.coef_)
# 截距
print("intercept:", model.intercept_)

# 测试数据
x_text = [[102, 4]]
predict = model.predict(x_text)
print("predict:", predict)

x1 = x_data[:, 0]
x2 = x_data[:, 1]
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x1, x2, y_data, color='g', marker='o')

#
x1 ,x2 = np.meshgrid(x1, x2)
z = model.intercept_ + x1 * model.coef_[0] + x2 * model.coef_[1]
# 画回归面
ax.plot_surface(x1, x2, z)


plt.show()