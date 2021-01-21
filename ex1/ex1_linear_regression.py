import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 单变量线性回归
path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
print(data.head())
print(data.describe())
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))
plt.show()


# 使用梯度下降实现线性回归
# Cost Function 代价函数
def computeCost(X, y, theta):
    # h(theta) = X * theta.T
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# 在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
# insert(loc列索引, column列标签, value列的值)
data.insert(0, 'Ones', 1)
# set X (training data) and y (target variable)
# shape[1] 表示data第二个维度的维数(即列数)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]  # X是所有行，去掉最后一列，即训练集
y = data.iloc[:, cols - 1:cols]  # y是所有行，最后一列，即目标变量

print(X.head())
print(y.head())

X = np.mat(X.values)
y = np.mat(y.values)
theta = np.mat(np.array([0, 0]))
print(theta)
print(X.shape, theta.shape, y.shape)
print(computeCost(X, y, theta))


# Batch Gradient Descent 批量梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    # np.zeros() 给定形状的用0填充的数组
    temp = np.mat(np.zeros(theta.shape))
    # theta.ravel() 降维
    # theta [[0, 0]] 降维后仍是[[0, 0]], shape[1] = 2
    # parameters = 2
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost


# 初始化一些附加变量 - 学习速率α和要执行的迭代次数
alpha = 0.01
iters = 1000

# 运行梯度下降算法来将我们的参数θ适合于训练集
g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g)
print(computeCost(X, y, g))

# 绘制线性模型以及数据，直观地看出它的拟合
# np.linspace() 在指定间隔内返回指定数量的均匀间隔的样本
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# 线性回归模型f
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

# 多变量线性回归
path2 = 'ex1data2.txt'
data2 = pd.read_csv(path2, header=None, names=['Size', 'Bedrooms', 'Price'])
print(data2.head())

# 特征归一化
data2 = (data2 - data2.mean()) / data2.std()
print(data2.head())

# 与单变量线性回归类似
# 添加一列全为1的值
data2.insert(0, 'Ones', 1)

# 设置X和y
cols2 = data2.shape[1]
X2 = data2.iloc[:, 0:cols2 - 1]
y2 = data2.iloc[:, cols2 - 1:cols2]

# 转化成Numpy矩阵，初始化theta
X2 = np.mat(X2.values)
y2 = np.mat(y2.values)
theta2 = np.mat(np.array([0, 0, 0]))

# 运行梯度下降算法选择合适的theta
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
# 运行损失函数
print(computeCost(X2, y2, g2))

# 可视化训练进程
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


# 使用Normal Equation(正规方程)进行线性回归
def normalEqn(X, y):
    #  np.linalg.inv() 矩阵求逆
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # X.T@X 等价于 X.T.dot(X) 点积
    return theta


final_theta = normalEqn(X, y)
print(final_theta)  # 单变量线性回归正规方程法求出的theta
print(g)  # 单变量线性回归梯度下降法求出的theta
