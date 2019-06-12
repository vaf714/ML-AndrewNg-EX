# 吴恩达机器学习-课后练习

使用 python3.7 编写，可以直接运行，**目前仍在整理中 ...**

## 导航
- [ex1 - 单变量线性回归](ex1_linear-regression/single_var.py)
    - **问题**: 使用一个变量实现线性回归，以预测餐厅的利润。假设您是一家餐厅的CEO，并正考虑在不同的城市开设新店。该连锁店已在各个城市拥有连锁店，您可以获得这些城市的利润和人口数据。使用此数据来帮助您选择要扩展的下一个城市。
    
    - **训练数据**: [`ex1data1.txt`](ex1_linear-regression/training_data/ex1data1.txt) 第一列是城市的人口，第二列是该城市的连锁店的利润。负值表示亏损。

    - **其他求解方法**：[正规方程求解](ex1_linear-regression/signle_var_use_normaleqn.py)、[使用 sklearn 模块](ex1_linear-regression/single_var_use_sklearn.py)
    
- [ex1 - 多变量线性回归](ex1_linear-regression/mult_var.py)

    - **问题**: 使用两个变量实现线性回归，预测房屋价格。
    
    - **训练数据**: [`ex1data2.txt`](ex1_linear-regression/training_data/ex1data2.txt) 第一列是房子的大小，第二列是卧室的数量，第三列是房屋的价格。

- [ex2 - 逻辑回归](ex2_logistic-regression/logistic_regression.py)
    - **问题**: 我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。设想你是大学相关部分的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。现在你拥有之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，你有他们两次测试的评分和最后是被录取的结果。为了完成这个预测任务，我们准备构建一个可以基于两次测试评分来评估录取可能性的分类模型。
    
    - **训练数据**: [`ex2data1.txt`](ex2_logistic-regression/training_data/ex2data1.txt) 第一列和第二列是两次测试的成绩，第三列表示是否录取。

- [ex2 - 正则化逻辑回归](ex2_logistic-regression/regularization.py)
    - **问题**: 设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。对于这两次测试，你想决定是否芯片要被接受或抛弃。为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，从其中你可以构建一个逻辑回归模型。
    
    - **训练数据**: [`ex2data2.txt`](ex2_logistic-regression/training_data/ex2data2.txt) 第一列和第二列是两次测试的数据，第三列表示是被接受。





## 参考
[Coursera-ML-AndrewNg-Notes](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes)