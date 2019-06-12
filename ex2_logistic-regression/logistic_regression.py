import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt


def init_data():
    data = pd.read_csv('training_data/ex2data1.txt', header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    # 插入一列便于向量化
    data.insert(0, 'Ones', 1)

    col_num = data.shape[1]
    X = data.iloc[:, 0:col_num - 1]
    y = data.iloc[:, col_num - 1:col_num]
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(X.shape[1])

    return data, X, y, theta


# 画图
def draw(data, theta):
    # 画出训练数据散点图
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]
    plt.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    plt.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    plt.legend()

    # 画出拟合函数
    x = np.linspace(data['Exam 1'].min(), data['Exam 1'].max())
    y = (-theta[0] - theta[1] * x) / theta[2]
    plt.plot(x, y, 'y', label='Prediction')

    plt.show()


# S型函数
def sigmoid(X, theta):
    return 1 / (1 + np.exp(-X * theta.T))


# 代价函数
def cost_func(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = -y.T * np.log(sigmoid(X, theta))
    second = (1 - y).T * np.log(1 - sigmoid(X, theta))

    return (first - second) / len(X)


# 一次梯度下降
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    return ((sigmoid(X, theta) - y).T * X) / len(X)


# 对预测结果分类
def predict(theta, X):
    X = np.matrix(X)
    theta = np.matrix(theta)
    probability = sigmoid(X, theta)
    return [1 if x >= 0.5 else 0 for x in probability]


# 计算拟合精度
def cal_precision(X, y):
    predictions = predict(result[0], X)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
    accuracy = (sum(map(int, correct)) % len(correct))
    return accuracy


if __name__ == '__main__':
    data_g, X_g, y_g, theta_g = init_data()
    # 拟合
    result = opt.fmin_tnc(func=cost_func, x0=theta_g, fprime=gradient, args=(X_g, y_g))
    # 画出训练数据散点图和拟合函数图
    draw(data_g, result[0])
    print('拟合精度: ', cal_precision(X_g, y_g), '%')

