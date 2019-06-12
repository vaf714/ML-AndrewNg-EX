# 正则化
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


def load_data():
    data = pd.read_csv('training_data/ex2data2.txt', header=None, names=['Test 1', 'Test 2', 'Accepted'])
    return data


# 画图
def draw(data):
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]
    plt.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
    plt.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Not Accepted')
    plt.legend()
    plt.show()


def init_data(data):
    degree = 5
    # 用两列变量构造出新的10列，用于构造多项式，拟合出复杂的模型
    for i in range(1, degree):
        for j in range(0, i):
            data['F' + str(i) + str(j)] = np.power(data['Test 1'], i - j) * np.power(data['Test 2'], j)

    # 删除原来的两列(axis=1 对列操作，inplace=True 在原来的矩阵操作)
    data.drop(['Test 1', 'Test 2'], axis=1, inplace=True)

    # 添加一列1便于向量化
    data.insert(1, 'Ones', 1)

    cols_num = data.shape[1]
    X = data.iloc[:, 1:cols_num]
    y = data.iloc[:, 0:1]
    X = np.array(X.values)
    y = np.array(y.values)
    theta = np.zeros(X.shape[1])
    # print(X.head(), y.head())
    return X, y, theta


# s型函数
def sigmoid(X, theta):
    return 1 / (1 + np.exp(-X * theta.T))


# 代价函数
def cost(theta, X, y, learning_rate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    m = len(X)
    first = -y.T * np.log(sigmoid(X, theta))
    second = (1 - y).T * np.log(1 - sigmoid(X, theta))
    reg = np.sum(np.power(theta[:, 1:theta.shape[1]], 2))   # 不对 theta0 进行惩罚

    return (1 / m) * (first - second) + (learning_rate / (2 * m)) * reg


# 一次梯度下降
def gradient_reg(theta, X, y, learning_rate):
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)

    m = len(X)
    parameter_num = theta.shape[1]  # theta 个数
    grad = np.zeros(parameter_num)

    error = sigmoid(X, theta) - y
    for i in range(parameter_num):
        if i == 0:
            # 正则化时theta0 不进行惩罚，单独更新
            temp = error.T * X[:, i]
        else:
            temp = error.T * X[:, i] + learning_rate * theta[:, i]
        grad[i] = temp / m

    return grad


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
    data_g = load_data()
    # 查看训练数据形状
    draw(data_g)
    # 初始化参数
    X_g, y_g, theta_g = init_data(data_g)

    # 开始拟合
    learning_rate_g = 1
    result = opt.fmin_tnc(func=cost, x0=theta_g, fprime=gradient_reg, args=(X_g, y_g, learning_rate_g))
    print('拟合精度: ', cal_precision(X_g, y_g), '%')
