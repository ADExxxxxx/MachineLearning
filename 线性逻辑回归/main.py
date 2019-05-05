import numpy as np
import matplotlib.pyplot as plt
# 该模块计算准确率，召回率等进行模型评估
from sklearn.metrics import classification_report
# 该模块用于标准化数据
from sklearn import preprocessing
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


# 数据处理添加偏置单元
y_data = y_data.reshape(100, 1)
X = np.concatenate((np.ones((100, 1)), x_data), axis=1)
print("原始的X数据:x1,x2的维度\n", x_data.shape)
print("原始的y的标签值的维度\n", y_data.shape)
print("添加偏置x0后的x维度", X.shape)


# 定义激活函数
# 1/(1 + e^-x)
def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


# 定义代价函数
# Cost(hθ(x), y) = -y*log(hθ(x))-(1-y)log(1-hθ(x))
def cost(xMat, yMat, ws):
    left = np.multiply(yMat, np.log(sigmoid(xMat*ws)))
    right = np.multiply(1 - yMat, np.log(1 - sigmoid(xMat*ws)))
    return np.sum(left + right) / -(len(xMat))


# 定义梯度下降函数
def gradient_descent(xArr, yArr):
    # 进行数据标准
    if scale == True:
        xArr = preprocessing.scale(xArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # print(xMat)
    lr = 1      # 学习率
    epochs = 10000  # 迭代次数
    costList = []   # 保存迭代过程中的cost值
    # 计算行列数
    # 行代表数据样本个数，列代表特征（权值）个数
    m, n = np.shape(xMat)
    print(m, n)
    # 初始化权值
    ws = np.mat(np.ones((n, 1)))
    # print(ws)
    for i in range(epochs+1):
        # 预测函数
        h = sigmoid(xMat * ws)
        # print(h)
        # 计算误差
        ws_grad = xMat.T * (h - yMat) / m
        ws = ws - lr * ws_grad
        if i % 50 == 0:
            # print(cost(xMat, yMat, ws))
            costList.append(cost(xMat, yMat, ws))
    # 注意缩进！注意缩进！ 一个缩进死全家
    return ws, costList


# 梯度下降进行计算
ws, costList = gradient_descent(X, y_data)
print(ws, costList)
# 数据未处理时画出线，若标准化，图像线会非常奇怪所以不予以画图
if scale == False:
    # 画出决策边界
    plot()
    # 通过两点确定一条直线
    x_test = [[-10], [10]]
    y_test = (-ws[0] - x_test * ws[1])/ws[2]
    plt.plot(x_test, y_test, 'k')
    plt.show()


# 画图观察loss的情况
x = np.linspace(0, 10000, 201)
plt.plot(x, costList, c='r')
plt.title("Train")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()


# 通过拟合的模型进行结果预测
def predict(x_data, ws):
    if scale == True:
        x_data = preprocessing.scale(x_data)
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    # 返回1或0，在结果>=0.5时为1 否则为0
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat*ws)]


predictions = predict(X, ws)
# 打印最终报告
print(classification_report(y_data, predictions))