# -*- coding: UTF-8 -*-

import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import max_error,mean_absolute_error,mean_squared_error,mean_squared_log_error,median_absolute_error,\
    r2_score,explained_variance_score,mean_tweedie_deviance,mean_poisson_deviance,mean_gamma_deviance

# 不使用 plt ，而是使用稍微底层点儿的API 来画图，有利于理解
class LinearRegressionFigure:
    lossfuns = {
        "Max Error": max_error,
        "Mean Absolute Error":  mean_absolute_error,
        "Mean Squared Error": mean_squared_error,
        "mean Squared Log Error":mean_squared_log_error,
        "Median Absolute Error":median_absolute_error,
        "R2 Score":r2_score,
        "Explained Variance Score":explained_variance_score,
        "Mean Tweedie Deviance":mean_tweedie_deviance,
        "Mean Poisson Deviance":mean_poisson_deviance,
        "Mean Gamma Deviance":mean_gamma_deviance,
    }
    def __init__(self, algorithms_num, cell_height = 4, cell_width = 4):
        # 一种算法，2*2 的分布图，后面 2*5 画 10 中 loss 函数，有些没有记录的就不画
        self.algorithms_num = algorithms_num                        # 有几种算法
        self.algorithms_idx = 0                                     # 当前算法的下标
        self.cell_cols = 7                                          # 格子数列数
        self.cell_rows = 2*self.algorithms_num                      # 垂直格行数
        self.cell_height, self.cell_width = cell_height,cell_width  # 单个格子高度和宽度

        # 创建画布，大小为 figsize = (width, height)
        self.fig = plt.figure(figsize=(self.cell_cols * self.cell_width, self.cell_rows * self.cell_width))
        # 创建一个网络布局
        self.gs = gridspec.GridSpec(self.cell_rows, self.cell_cols, self.fig)

        figure_size = ()
    def draw(self, X_train, X_test, Y_train, Y_test, Y_predict, records, title, xlabel = 'x', ylabel = 'y'):
        # 画分散点布图
        # plt.subplot2grid((self.cell_rows, self.cell_cols), (2 * self.algorithms_idx, 0), colspan=2, rowspan=2)
        # plt.scatter(X_train[:,0], Y_train, marker="o", color="k", label="Real Train")
        # plt.scatter(X_test[:,0], Y_test, marker="s", color="r", label="Real Test")
        # plt.plot(X_test[:,0], Y_predict, color="b", label="Predict")
        # plt.title(title)
        # plt.xlabel(xlabel)
        # plt.ylabel(ylabel)
        # plt.legend()
        subplotspec = self.gs.new_subplotspec((2 * self.algorithms_idx, 0), rowspan=2, colspan=2)
        ax = self.fig.add_subplot(subplotspec)
        ax.scatter(X_train[:,0], Y_train, marker="o", color="k", label="Real Train")
        ax.scatter(X_test[:,0], Y_test, marker="s", color="r", label="Real Test")
        ax.plot(X_test[:,0], Y_predict, color="b", label="Predict")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        # loss
        cnt = 0
        for name, fun in LinearRegressionFigure.lossfuns.items():
            # plt.subplot2grid((self.cell_rows, self.cell_cols), (2 * self.algorithms_idx + int(cnt/5), 2 + cnt%5), colspan=1, rowspan=1)
            # # loss，中途训练集优化记录的
            # if records and 'epochs' in records and len(records['epochs']) > 0:
            #     plt.plot(records['epochs'], [(fun(Y_train, predict)) for predict in records['predicts']], color="b", label=name)
            # # loss，最终测试集的
            # plt.scatter([0], [fun(Y_test, Y_predict)], marker="s", color="r", label="Predict")
            # plt.annotate('%.3f' % fun(Y_test, Y_predict), (0, fun(Y_test, Y_predict)))
            # plt.title(name)
            # plt.xlabel('epochs')
            # plt.ylabel('loss')
            # plt.legend()
            # cnt+=1

            subplotspec = self.gs.new_subplotspec((2 * self.algorithms_idx + int(cnt/5), 2 + cnt%5), rowspan=1, colspan=1)
            ax = self.fig.add_subplot(subplotspec)
            # loss，中途训练集优化记录的
            if records and 'epochs' in records and len(records['epochs']) > 0:
                ax.plot(records['epochs'], [(fun(Y_train, predict)) for predict in records['predicts']], color="b", label=name)
            # loss，最终测试集的
            ax.scatter([0], [fun(Y_test, Y_predict)], marker="s", color="r", label="Predict")
            ax.annotate('%.3f' % fun(Y_test, Y_predict), (0, fun(Y_test, Y_predict)))
            ax.set_title(name)
            ax.set_xlabel('epochs')
            ax.set_ylabel('loss')
            ax.legend()
            cnt+=1
        
        self.algorithms_idx = self.algorithms_idx + 1
    
    def show(self):
        # plt.tight_layout()
        # plt.show()
        self.fig.tight_layout(pad=2)
        self.fig.show()

    def save(self, fname):
        # plt.savefig(fname)
        self.fig.savefig(fname, bbox_inches='tight')
