# -*- coding: UTF-8 -*-

import numpy as np
import sklearn
from sklearn import linear_model
# from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def _preprocess_data(X):
    # 插入 x0 = 1 偏置项
    X = np.insert(X, 0, 1, axis = 1)
    return X

def make_regression(n_samples=100, n_features=100, n_targets=1, bias=0.0, noise=0.0, coef=False, random_state=None):
    generator = np.random.RandomState(random_state)
    X = generator.randn(n_samples, n_features)
    # 全部转为正数
    X = X - X.min()
    
    ground_truth = 100 * generator.rand(n_features, n_targets)

    Y = np.dot(X, ground_truth) + bias

    # Add noise
    if noise > 0.0:
        Y += generator.normal(scale=noise, size=Y.shape)

    Y = np.squeeze(Y)
    
    if coef:
        return X, Y, np.squeeze(ground_truth)

    else:
        return X, Y

class LinearRegression:
    def __init__(self, algorithm, *args, **kwargs):
        algorithms = {
            'ols': self.ols,
            'bgd': self.bgd,
            'sgd': self.sgd,
            'mbgd': self.mbgd,
        }
        self.algorithmfun = algorithms[algorithm]
        self.args = args
        self.kwargs = kwargs
        self.records = {
            'epochs': [],
            'predicts': [],
        }
    
    def _get_kw_arg(self, k, default):
        if k in self.kwargs:
            return self.kwargs[k]
        return default

    def _set_intercept(self, _Theta):
        self.intercept_  = _Theta[0]
        self.coef_ = _Theta[1:]

    # 添加记录，记录优化过程
    def add_record(self, epoch, X, _Theta):
        if self._get_kw_arg('record_predict', False):
            self._set_intercept(_Theta)
            self.records['epochs'].append(epoch)
            self.records['predicts'].append(self.predict(X))

    def _max_epochs(self):
        return self._get_kw_arg('max_epochs', 1000)
    
    def _learning_rate(self):
        return  self._get_kw_arg('learning_rate', 0.005)

    def _batch_size(self):
        return  self._get_kw_arg('batch_size', 10)

    def fit(self, X, Y):
        return self.algorithmfun(X, Y)

    def predict(self, X):
        # 根据 (1) 式子来计算
        # 偏置项没有在 coef_ 中，x0 = 1 ，这里直接加上 intercept_ 即可
        return X.dot(self.coef_) + self.intercept_
    
    # 普通最小二乘法（Ordinary Least Squares, OLE)
    def ols(self, X, Y):
        X = _preprocess_data(X)

        # 根据式子 (5) 计算 \Theta
        XTX = X.T.dot(X)

        if np.linalg.det(XTX) == 0.0 :
            print("The matrix connot do inverse")
            return
        
        _Theta = np.linalg.inv(XTX).dot(X.T).dot(Y)
        self._set_intercept(_Theta)
    
    # 批量梯度下降（Batch Gradient Descent，BGD）
    def bgd(self, X, Y):
        rawX = X.copy()
        X = _preprocess_data(X)

        # 迭代次数
        max_epochs = self._max_epochs()
        # 学习率
        learning_rate = self._learning_rate()

        # m 为样本数， n 为特征数（这里是原始特征数 + 1个偏置项）
        (m, n) = X.shape

        # 随机初始化 _Theta
        _Theta = np.random.rand(n)

        # 迭代次数
        for epoch in range(max_epochs):
            # 每次使用所有的样本
            
            # 使用矩阵形式来计算
            gradients = 1/m * X.T.dot(X.dot(_Theta) - Y)
            _Theta = _Theta - learning_rate * gradients
            
            # 记录下 record
            self.add_record(epoch, rawX, _Theta)
        
        self._set_intercept(_Theta)
    
    # 随机梯度下降（Stochastic Gradient Descent，SGD）
    def sgd(self, X, Y):
        rawX = X.copy()
        X = _preprocess_data(X)

        # 迭代次数
        max_epochs = self._max_epochs()
        # 学习率
        learning_rate = self._learning_rate()

        # m 为样本数， n 为特征数（这里是原始特征数 + 1个偏置项）
        (m, n) = X.shape

        # 随机初始化 _Theta
        _Theta = np.random.rand(n)

        # 迭代次数
        for epoch in range(max_epochs):
            # 随机以下，每次使用1个样本
            random_indeies = [x for x in range(m)]
            np.random.shuffle(random_indeies)
            for i in range(0, m):
                random_X = X[i:i+1]
                random_Y = Y[i:i+1]

                # 使用矩阵形式来计算
                gradients = random_X.T.dot(random_X.dot(_Theta) - random_Y)
                _Theta = _Theta - learning_rate * gradients
            
            # 记录下 record
            self.add_record(epoch, rawX, _Theta)
        
        self._set_intercept(_Theta)

    # 小批量梯度下降（Mini-Batch Gradient Descent, MBGD）
    def mbgd(self, X, Y):
        rawX = X.copy()
        X = _preprocess_data(X)

        # 迭代次数
        max_epochs = self._max_epochs()
        # 学习率
        learning_rate = self._learning_rate()
        # 批量的大小
        batch_size = self._batch_size()

        # m 为样本数， n 为特征数（这里是原始特征数 + 1个偏置项）
        (m, n) = X.shape

        # 随机初始化 _Theta
        _Theta = np.random.rand(n)

        # 迭代次数
        for epoch in range(max_epochs):
            # 随机以下，每次使用batch_size个样本
            random_indeies = [x for x in range(m)]
            np.random.shuffle(random_indeies)
            for i in range(0, m, batch_size):
                random_X = X[i:i+batch_size]
                random_Y = Y[i:i+batch_size]

                gradients = random_X.T.dot(random_X.dot(_Theta) - random_Y)
                _Theta = _Theta - learning_rate * gradients
            
            # 记录下 record
            self.add_record(epoch, rawX, _Theta)
        
        self._set_intercept(_Theta)

def main():
    # 截距
    intercept = 100.0
    # 生成随机数字
    X, Y, coef = make_regression(n_samples=100, n_features=1,bias=intercept,noise=10, coef=True,random_state=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

    # # 画图相关的
    from LinearRegressionFigure import LinearRegressionFigure
    lrf = LinearRegressionFigure(6)

    # 使用 sklearn 来训练
    lr = sklearn.linear_model.LinearRegression()
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    lrf.draw(X_train, X_test, Y_train, Y_test, Y_predict, None, 'sklearn LinearRegression')

    lr = LinearRegression('ols', record_predict = True)
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    lrf.draw(X_train, X_test, Y_train, Y_test, Y_predict, lr.records, 'Ordinary Least Squares, OLE')

    lr = LinearRegression('bgd', record_predict = True)
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    lrf.draw(X_train, X_test, Y_train, Y_test, Y_predict, lr.records, 'Batch Gradient Descent, BGD')

    lr = LinearRegression('sgd', record_predict = True)
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    lrf.draw(X_train, X_test, Y_train, Y_test, Y_predict, lr.records, 'Stochastic Gradient Descent, SGD')

    lr = sklearn.linear_model.SGDRegressor()
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    lrf.draw(X_train, X_test, Y_train, Y_test, Y_predict, None, 'sklearn SGDRegressor')

    lr = LinearRegression('mbgd', record_predict = True)
    lr.fit(X_train, Y_train)
    Y_predict = lr.predict(X_test)

    lrf.draw(X_train, X_test, Y_train, Y_test, Y_predict, lr.records, 'Mini-Batch Gradient Descent, MBGD')
    
    # 显示图
    lrf.show()
    lrf.save("figure.png")

if __name__ == "__main__":
    main()