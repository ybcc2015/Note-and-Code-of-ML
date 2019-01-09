# coding: utf-8
import numpy as np
from sklearn.datasets import load_iris


def get_iris_data():
    """
    加载Iris数据集（只取前两个特征及前两个类别）
    """
    iris = load_iris()
    X = iris.data[:, :2][:100]
    Y = iris.target[:100]
    # 将类'0'转换成'-1'
    Y = np.where(Y, 1, -1)

    assert (X.shape == (100, 2))
    assert (Y.shape == (100,))
    return X, Y
