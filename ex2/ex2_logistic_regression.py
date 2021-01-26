import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

'''
在训练的初始阶段，我们将要构建一个逻辑回归模型来预测，某个学生是否被大学录取。
设想你是大学相关部分的管理者，想通过申请学生两次测试的评分，来决定他们是否被录取。
现在你拥有之前申请学生的可以用于训练逻辑回归的训练样本集。对于每一个训练样本，你有他们两次测试的评分和最后是被录取的结果。
为了完成这个预测任务，我们准备构建一个可以基于两次测试评分来评估录取可能性的分类模型。
'''
path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(data.head())

# 创建分数的散点图，正样本（被录取）/ 负样本（未被录取）
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()


# 常用的逻辑函数：sigmoid函数，S型函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 检查sigmoid函数是否正常工作
nums = np.arange(-10, 10, step=1)

fix, ax = plt.subplots(figsize=(12, 8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


# Cost Function
def cost(theta, X, y):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    first = np.multiply(y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return -np.sum(first + second) / len(X)


# 一些初始化设置
# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert to numpy arrays and initialize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

print(theta)
print(X.shape, theta.shape, y.shape)
print(cost(theta, X, y))


# 梯度下降
# 实际上没有在这个函数中执行梯度下降，仅仅在计算一个梯度步长
def gradient(theta, X, y):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        grad[i] = np.sum(term) / len(X)

    return grad


# 用我们的数据和初始参数为0的梯度下降法的结果
print(gradient(theta, X, y))

# 用SciPy's truncated newton（TNC）实现寻找最优参数
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, y))
print(result)

# 这个结论下Cost函数的计算结果
print(cost(result[0], X, y))


# 预测函数：我们需要编写一个函数，用我们所学的参数theta来为数据集X输出预测。
# 然后，我们可以使用这个函数来给我们的分类器的训练精度打分
def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


theta_min = np.mat(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))


'''
在训练的第二部分，我们将要通过加入正则项提升逻辑回归算法。
如果你对正则化有点眼生，或者喜欢这一节的方程的背景，请参考在"exercises"文件夹中的"ex2.pdf"。
简而言之，正则化是成本函数中的一个术语，它使算法更倾向于“更简单”的模型（在这种情况下，模型将更小的系数）。
这个理论助于减少过拟合，提高模型的泛化能力。这样，我们开始吧。

设想你是工厂的生产主管，你有一些芯片在两次测试中的测试结果。对于这两次测试，你想决定是否芯片要被接受或抛弃。
为了帮助你做出艰难的决定，你拥有过去芯片的测试数据集，从其中你可以构建一个逻辑回归模型。
'''
path = 'ex2data2.txt'
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])
print(data2.head())

positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
