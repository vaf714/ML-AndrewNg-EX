# 公用模块
import numpy as np


# 初始化矩阵
def init_matrix(data):
    # 初始化数据
    data.insert(0, "Ones", 1)   # 添加一列便于向量化
    cols = data.shape[1]    # 列数
    X = data.iloc[:, 0:cols - 1]
    y = data.iloc[:, cols-1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)
    return X, y


# 初始化theta
def init_theta(data):
    # 初始化数据
    cols = data.shape[1]    # 列数
    theta = np.matrix(np.zeros(cols - 1))
    return theta


# 计算代价
def compute_cost(X, y, theta):
    inner = np.power(X * theta.T - y, 2)
    return sum(inner) / (2 * X.shape[0])


# 梯度下降
def gradient_decent(X, y, theta, alpha, step):
    cost_list = np.zeros(step)  # 记录每一次迭代的代价

    for i in range(step):
        error = X * theta.T - y
        # 更新theta
        theta = theta - ((alpha / X.shape[0]) * (error.T * X))

        cost_list[i] = compute_cost(X, y, theta)
    return theta, cost_list