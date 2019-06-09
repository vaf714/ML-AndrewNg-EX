# 单变量线性回归方法三: 使用正规方程计算
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from base import init_matrix


# 读取数据
data = pd.read_csv("training_data/ex1data1.txt", header=None, names=['Population', 'Profit'])

# 初始化数据
X, y = init_matrix(data)

# 拟合
theta = (X.T * X).I * X.T * y
print(theta)

# 画图
x = np.linspace(data.Population.min(), data.Population.max())
y = theta[0, 0] + (theta[1, 0] * x)
plt.plot(x, y, 'r', label='Prediction')
plt.scatter(data.Population, data.Profit, label='Traning Data')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population Size')
plt.legend()
plt.show()