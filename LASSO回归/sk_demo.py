"""
这里不给出原始公式推导，因为相比较岭回归，LASSO回归的区别不大
二者之间主要是正则化项不相同
在代价函数方面：
    岭回归：1/(2*m)[sum(i=1->m)(hθ(xi)-yi)^2]+λ*sum(j=0->n)θj^2
    LASSO回归:1/(2*m)[sum(i=1->m)(hθ(xi)-yi)^2]+λ*sum(j=0->n)|θj|
LASSO回归具有更强的解释性，他将与其他特征线性相关的特征系数置于0
详细数学细节不予赘述
"""

import numpy as np
from sklearn import linear_model

# 读入数据
data = np.genfromtxt("longley.csv", delimiter=",")
x_data = data[1:, 2:]
y_data = data[1:, 1]

# 训练LASSO
model = linear_model.LassoCV()
model.fit(x_data, y_data)

# lasso 系数
print("Lasso系数: ", model.alpha_)
# 相关系数
# 这里打印出的系数有0说明存在多重共线性
print("相关系数: ", model.coef_)

# 做一个预测
predict = model.predict(x_data[-2, np.newaxis])
print(predict)
