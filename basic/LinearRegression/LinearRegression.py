# -*- coding: UTF-8 -*-

import numpy as np

class LinearRegression:
    def __init__(self):
        self._Theta = None
    
    def fit(self, X, Y):
        # 插入 x0 = 1
        X = np.insert(X, 0, 1, axis = 1)
        # 根据式子 (5) 计算 \Theta
        XTX = X.T.dot(X)

        if np.linalg.det(XTX) == 0.0 :
            print("The matrix connot do inverse")
            return
        
        self._Theta = np.linalg.inv(XTX).dot(X.T).dot(Y)
    
    def predict(self, X):
        # 插入 x0 = 1
        X = np.insert(X, 0, 1, axis = 1)
        # 根据 (1) 式子来计算
        return X.dot(self._Theta)

if __name__ == "__main__":
    import matplotlib.pylab as plt
    from sklearn import datasets
    from sklearn.metrics import mean_squared_error

    # 导入数据集
    diabetes = datasets.load_diabetes()

    # 为了画图方便，这里只取2个特征
    x_train, x_test = diabetes.data[:-20,:2],diabetes.data[-20:,:2]
    y_train, y_test = diabetes.target[:-20],diabetes.target[-20:]

    clf = LinearRegression()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)

    print("Mean Squared Error", mean_squared_error(y_test, y_predict))

    plt.scatter(x_test[:,0], y_test, color="k")
    plt.plot(x_test[:,0], y_predict, color="b")
    plt.show()
