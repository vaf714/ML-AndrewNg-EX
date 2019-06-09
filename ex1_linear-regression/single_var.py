# 单个变量
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# 读取数据
path = "ex1data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

# 观察数据形状
# data.plot(kind='scatter', x='Population', y='Profit')
# plt.show()

# 初始化数据
data.insert(0, "Ones", 1)   # 添加一列便于向量化
cols = data.shape[1]    # 列数
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols-1:cols]
# print(X.head())
# print(y.head())

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

alpha = 0.01
step = 1000

# 拟合
g, cost = gradient_decent(X, y, theta, alpha, step)
print('误差: ', compute_cost(X, y, g))

# 绘制拟合图像
x = np.linspace(data.Population.min(), data.Population.max())
y = g[0, 0] + (g[0, 1] * x)
plt.plot(x, y, 'r', label='Prediction')
plt.scatter(data.Population, data.Profit, label='Traning Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population Size')
plt.legend()
plt.show()

# 绘制损失函数图像
plt.plot(range(step), cost)
plt.show()
