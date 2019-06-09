# 多变量，梯度下降方法
import numpy as np
import pandas as pd
from base import compute_cost, gradient_decent, init_matrix, init_theta


# 读取数据
data = pd.read_csv("training_data/ex1data2.txt", header=None, names=['Size', 'Bedrooms', 'Price'])

# 归一化处理，使用 Z-score 标准化方法
data = (data - data.mean()) / data.std()

# 初始化数据
X, y = init_matrix(data)
theta = init_theta(data)

# 梯度下降求解
alpha = 0.01
step = 1000
g, cost = gradient_decent(X, y, theta, alpha, step)
print('误差: ', compute_cost(X, y, g))
