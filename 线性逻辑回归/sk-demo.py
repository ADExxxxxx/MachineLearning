import numpy as np
import matplotlib.pyplot as plt
# 该模块计算准确率，召回率等进行模型评估
from sklearn.metrics import classification_report
# 该模块用于标准化数据
from sklearn import preprocessing
from sklearn import linear_model
scale = False   # 是否进行数据标准化

data = np.genfromtxt("data.csv", delimiter=',')
print(data)
x_data = data[:, :-1]     # 特征值
y_data = data[:, -1]        # 标签值


# 划分a，b类，并且分成两个散点图
def plot():
    x0 = []
    y0 = []     # a类
    x1 = []
    y1 = []     # b类
    # 切分不同类别的数据
    for i in range(len(x_data)):
        if y_data[i] == 0:
            x0.append(x_data[i, 0])
            y0.append(x_data[i, 1])
        else:
            x1.append(x_data[i, 0])
            y1.append(x_data[i, 1])

    # 画图
    scatter0 = plt.scatter(x0, y0, c='b', marker='o')
    scatter1 = plt.scatter(x1, y1, c='r', marker='x')
    # 画图例
    plt.legend(handles=[scatter0, scatter1], labels=['label0', 'label1'], loc='best')


# 尽力逻辑回归模型并进行拟合
logistic = linear_model.LogisticRegression()
logistic.fit(x_data, y_data)


if scale == False:
    plot()
    x_test = np.array([[-4], [3]])
    # 此公式与原始含义一样
    # intercept_:偏置
    # coef_:权重
    y_test = (-logistic.intercept_ - x_test*logistic.coef_[0][0]) / logistic.coef_[0][1]
    plt.plot(x_test, y_test, 'k')
    plt.show()
prediction = logistic.predict(x_data)
print(classification_report(y_data, prediction))
