# coding: utf-8
import numpy as np
from helper import get_iris_data


class PLA(object):
    def __init__(self, lr=0.01, max_iter=1000, shuffle=True, log=True, log_step=50):
        """
        模型构造函数
        :param lr: 学习率
        :param max_iter: 最大迭代次数
        :param shuffle: 是否打乱训练集
        :param log: 是否打印日志
        :param log_step: 打印日志间隔
        """
        self.max_iter = max_iter
        self.lr = lr
        self.shuffle = shuffle
        self.log = log
        self.log_step = log_step
        self.w = None
        self.b = 0
    
    def sign(self, x):
        """
        sign函数
        :param x: number
        :return: 1 or -1
        """
        return 1 if x >= 0 else -1
    
    def fit(self, X, Y):
        """
        训练
        :param X: 2-D array, shape(num_samples, num_features), 包含了所有训练数据的feature
        :param Y: 1-D array, shape(num_samples,), 包含了所有训练样本的label
        :return: None
        """
        # 样本数量
        m = X.shape[0]
        # 初始化w
        self.w = np.random.randn(X.shape[1])
        # 打乱顺序
        if self.shuffle:
            idx = list(range(m))
            np.random.shuffle(idx)
            X = X[idx]
            Y = Y[idx]
        
        iter_count = 0
        while iter_count < self.max_iter:
            wrong_samples = []  # 用来存放误分类样本的Loss
            for i in range(m):
                x = X[i]
                y = Y[i]
                tmp = y * np.add(np.dot(self.w, x), self.b)  # y * (wx + b)
                if tmp <= 0:
                    # 更新w,b
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    wrong_samples.append(-tmp)

            # 全部分类正确，停止训练
            if not wrong_samples:
                print('iter_count: {}\tloss: 0.0'.format(iter_count))
                break
            
            if self.log:
                # 每迭代50次打印一次训练日志
                if iter_count % self.log_step == 0 and wrong_samples:
                    loss = np.sum(wrong_samples)
                    print('iter_count: {}\tloss: {:.4f}'.format(iter_count, loss))
                
            iter_count += 1
        print('training finished!')
            
    def predict(self, x):
        """
        预测
        :param x: 要预测的数据, shape(2,)
        :return: 1 or -1
        """
        value = np.dot(self.w, x) + self.b
        pred = self.sign(value)
        return pred


if __name__ == '__main__':
    X, Y = get_iris_data()
    model = PLA()
    model.fit(X, Y)
    # 预测
    test_pos_data = np.array([7.0, 3.5])  # 随机生成一个正样本
    test_neg_data = np.array([4.2, 2.5])  # 随机生成一个负样本
    print(model.predict(test_pos_data))
    print(model.predict(test_neg_data))
