import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from  sklearn.linear_model import LinearRegression

# 载入数据
data = np.genfromtxt("data.csv", delimiter=',')
x_data = data[1:, 1, np.newaxis]
y_data = data[1:, 2, np.newaxis]
plt.scatter(x_data, y_data)

model = LinearRegression()
model.fit(x_data, y_data)
plt.plot(x_data, y_data, 'r')
# 通过线性回归拟合效果比较差
plt.plot(x_data, model.predict(x_data), 'g')

# 多项式模型,degree=n, 一直到x的n次方,根据degree的大小可以模拟出能拟合的最高维度
poly_model = PolynomialFeatures(degree=3)
#特征处理
x_poly = poly_model.fit_transform(x_data)
print(x_poly)
# 定义回归模型
line_model = LinearRegression()
line_model.fit(x_poly, y_data)
plt.plot(x_data, line_model.predict(poly_model.fit_transform(x_data)), 'purple')
plt.show()