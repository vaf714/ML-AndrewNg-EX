# 单变量线性回归方法二: 使用 sklearn 模块计算
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from base import init_matrix


# 读取数据
data = pd.read_csv("training_data/ex1data1.txt", header=None, names=['Population', 'Profit'])

# 初始化数据
X, y = init_matrix(data)

# 拟合
model = linear_model.LinearRegression()
model.fit(X, y) # 分析模型参数
f = model.predict(X)    # 用上一步的模型对变量进行预测获得的值

# 画图
x = np.array(X[:, 1])
plt.plot(x, f, 'r', label='Prediction')
plt.scatter(data.Population, data.Profit, label='Traning Data')
plt.legend()
plt.xlabel('Population')
plt.ylabel('Profit')
plt.title('Predicted Profit vs. Population Size')
plt.show()
