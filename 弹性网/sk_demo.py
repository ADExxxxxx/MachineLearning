"""
弹性网结合了岭回归和LASSO回归的特点
既使用了L1正则化，也使用了L2正则化
正则化项: λ * sum(j = 1 -> n)(α * θ^2 + (1 - α)|θj|)
具有更加好的效果
"""
import numpy as np
from sklearn import linear_model

# 读入数据
data = np.genfromtxt("longley.csv", delimiter=",")
x_data = data[1:, 2:]
y_data = data[1:, 1]

# 训练LASSO
model = linear_model.ElasticNetCV()
model.fit(x_data, y_data)

# ElasticNet 系数
print("ElasticNet系数: ", model.alpha_)
# 相关系数
# 这里打印出的系数有0说明存在多重共线性
print("相关系数: ", model.coef_)

# 做一个预测
predict = model.predict(x_data[-2, np.newaxis])
print(predict)
