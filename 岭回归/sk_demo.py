import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


data = np.genfromtxt("longley.csv", delimiter=",")
x_data = data[1:, 2:]
y_data = data[1:, 1]
# 创建模型
# 生成值作为lamda的值作为岭回归系数
lamda_to_test = np.linspace(0.01, 1).reshape(50, 1)
print(lamda_to_test.shape)

# 建立岭回归的模型,并交叉验证
model = linear_model.RidgeCV(alphas=lamda_to_test, store_cv_values=True)
model.fit(x_data, y_data)
# 岭系数
print(model.alpha_)
# loss值
# (x1, x2),x1:数据数量,x2:岭系数个数
print(model.cv_values_.shape)
print(model.cv_values_.mean(axis=0).shape)
# 第一个图表示，x坐标为50个Lamda的值，y坐标表示的是loss的均值
plt.plot(lamda_to_test, model.cv_values_.mean(axis=0))

# 第二个图表示，取得最小loss时的岭系数的点
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)), 'ro')
plt.show()