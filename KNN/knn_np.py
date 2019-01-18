# coding: utf-8
import numpy as np
from sklearn.datasets import load_iris


class KNN(object):
    def __init__(self, X, Y, k_neighbors, p=2):
        self.k_neighbors = k_neighbors
        self.p = p
        self.X = X
        self.Y = Y
        assert (k_neighbors <= X.shape[0])

    def get_distance(self, x):
        if self.p == 1:
            return np.sum(np.abs(self.X - x), axis=1)
        elif self.p == 2:
            return np.sqrt(np.sum(np.power(self.X - x, 2), axis=1))
        elif self.p == 'inf':
            return np.max(np.abs(self.X - x), axis=1)

    def predict(self, x):
        distance = self.get_distance(x)
        # 返回距离从大到小对应的索引
        indx = np.argsort(distance)
        # 最近k个样本的类别
        nearest_k_labels = self.Y[indx[:self.k_neighbors]]
        # 找到出现频率最高的类别
        counts = np.bincount(nearest_k_labels)
        result = np.argmax(counts)

        return result, nearest_k_labels


if __name__ == '__main__':
    # 加载Iris数据集
    iris = load_iris()
    X = iris.data[:, :2]  # 只取前两个特征
    Y = iris.target

    knn_model = KNN(X, Y, k_neighbors=3)

    # 预测
    test_sample = np.array([6, 3.2])
    label, nearest_labels = knn_model.predict(test_sample)
    print('最邻近的训练样本类别为：', nearest_labels)
    print('预测类别为：', label)
