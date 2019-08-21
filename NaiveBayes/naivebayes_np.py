# coding: utf-8
import numpy as np
import pandas as pd

class NaiveBayes(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.shape[0]  # 训练集大小
        self.n = X.shape[1]  # 特征个数
        self.classes = np.unique(Y)  # 类别可取值
        self.K = self.classes.size  # 类别总数
        self.prior_prob = None
        self.cond_prob = None
        self.poster_prob = None

    def get_prior_prob(self):
        """
        计算每个类别的先验概率
        """
        prior_prob = {}
        for c in self.classes:
            _sum = np.sum(np.where(self.Y == c, 1, 0))
            key = 'Y={}'.format(c)
            prior_prob[key] = (_sum + 1) / (self.N + self.K)  # Laplace smoothing
        return prior_prob

    def get_conditional_prob(self):
        """
        计算每个特征值在特定类别下的条件概率
        """
        cond_prob = {}
        # 遍历每一类
        for c in self.classes:
            inds = np.where(self.Y == c)[0]
            datas = X[inds]  # 类别为c的所有样本
            # 遍历每个特征
            for j in range(self.n):
                a = X[:, j]
                values = np.unique(a)  # 特征a的可取值
                Sj = values.size  # 特征a的可取值数量
                for v in values:
                    _sum = np.sum(np.where(datas[:, j] == v, 1, 0))
                    key = 'X{}={}|Y={}'.format(j + 1, v, c)
                    cond_prob[key] = (_sum + 1) / (len(datas) + Sj)  # Laplace smoothing
        return cond_prob

    def fit(self):
        self.prior_prob = self.get_prior_prob()
        self.cond_prob = self.get_conditional_prob()

    def predict(self, x):
        poster_prob = {}
        for c in self.classes:
            prob = 1.0
            prior_key = 'Y={}'.format(c)
            prob *= self.prior_prob[prior_key]
            for j in range(self.n):
                cond_key = 'X{}={}|Y={}'.format(j + 1, x[j], c)
                if cond_key in self.cond_prob.keys():
                    cond_prob = self.cond_prob[cond_key]
                else:
                    cond_prob = 1 / (np.sum(np.where(self.Y == c, 1, 0)) + np.unique(self.X[:, j]).size)
                prob *= cond_prob
            poster_prob[c] = prob
        self.poster_prob = poster_prob
        cls = sorted(poster_prob.items(), key=lambda x: x[1], reverse=True)[0][0]
        return cls


if __name__ == '__main__':
    # load data
    df = pd.read_csv('datas/data4_1.csv')
    # transform to numpy array
    X = df[['X1', 'X2']].values
    Y = df['Y'].values

    # 训练
    model = NaiveBayes(X, Y)
    model.fit()

    # 预测
    x = np.array([2, 'S'])
    y = model.predict(x)
    print('后验概率:', model.poster_prob)
    print('预测结果:', y)
