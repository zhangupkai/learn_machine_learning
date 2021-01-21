import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(object):
    """Perceptron classifier. 感知器进行分类

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0) 学习率
    n_iter : int
      Passes over the training dataset. 训练次数
    random_state : int
      Random number generator seed for random weight
      initialization. 随机数的种子

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        # rgen: NumPy随机数生成器，随机种子由用户指定，因此可以保证在需要时重现以前的结果
        rgen = np.random.RandomState(self.random_state)
        # 产生标准差为0.01的正态分布
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])  # shape[1] X第二个维度的维数
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # 特征 和 目标标签
            for xi, target in zip(X, y):
                # 调用predict方法预测标签并更新权重
                update = self.eta * (target - self.predict(xi))
                # 更新权重: w1至wn
                self.w_[1:] += update * xi
                # 更新权重零w0，权重零定义为w0=-theta (theta为阈值)，对应的x0=1
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        # np.dot() 点积
        # 求出净输入z
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        # np.where(condition, a, b) 满足条件condition则输出a，不满足则输出y
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# Plotting the Iris data #
df = pd.read_csv('iris.data', header=None)
# print(df.tail())

# select setosa and versicolor
# 0:100 等价于 [0,100), 从0到99共100行数据
# 4 第4列的值(初始为第0列): Iris-setosa为1, Iris-versicolor为-1
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length(第0列) and petal length(第2列)
X = df.iloc[0:100, [0, 2]].values

# plot data
# :50 等价于 0:50, X[:50, 0]即矩阵X的前50行，第一列
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa(山鸢尾)')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor(变色鸢尾)')

plt.xlabel('sepal length(萼片长度) [cm]')
plt.ylabel('petal length(花瓣长度) [cm]')
plt.legend(loc='upper left')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# plt.savefig('images/02_06.png', dpi=300)
plt.show()


# Training the perceptron model #
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')

plt.xlabel('Epochs(迭代)')
plt.ylabel('Numbers of updates(更新次数)')
plt.show()


# A function for plotting decision regions #
# 二维数据决策边界函数
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # X[:, 0].min() 第一列的特征向量的最小值
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # np.meshgrid: 创建网格阵列xx1和xx2
    # np.arange(start, stop, step): 从数值范围创建数组
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # ravel() 将多维数组转化为一维数组
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    # 画出轮廓图
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length(萼片长度) [cm]')
plt.ylabel('petal length(花瓣长度) [cm]')
plt.legend(loc='upper left')
plt.show()
