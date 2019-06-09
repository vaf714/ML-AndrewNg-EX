# 单变量，梯度下降方法
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import compute_cost, gradient_decent, init_matrix, init_theta


# 读取数据
data = pd.read_csv("training_data/ex1data1.txt", header=None, names=['Population', 'Profit'])

# 观察数据形状
# data.plot(kind='scatter', x='Population', y='Profit')
# plt.show()

# 初始化数据，得到X, y, theta矩阵
X, y = init_matrix(data)
theta = init_theta(data)

# 梯度下降求解
alpha = 0.01
step = 1000
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
# plt.plot(range(step), cost)
# plt.show()
