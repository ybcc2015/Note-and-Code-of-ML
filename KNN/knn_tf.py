# coding: utf-8
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

# 加载Iris数据集
iris = load_iris()
X = iris.data[:, :2]  # 只取前两个特征
Y = iris.target

# tf计算图的输入
X_input = tf.placeholder(tf.float32, shape=(None, 2))
test_input = tf.placeholder(tf.float32, shape=(2,))

# L2距离
pred = tf.sqrt(tf.reduce_sum(tf.pow(X_input - test_input, 2), axis=1))

# 测试样本
test_sample = np.array([6, 3.2])

n_neighbors = 3

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    distance = sess.run(pred, feed_dict={X_input: X, test_input: test_sample})
    indx = np.argsort(distance)
    # 最近k个样本的类别
    nearest_k_labels = Y[indx[:n_neighbors]]
    # 计算类别出现频率
    counts = np.bincount(nearest_k_labels)
    # 预测结果
    label = np.argmax(counts)

    print('最邻近的训练样本类别为：', nearest_k_labels)
    print('预测类别为：', label)
