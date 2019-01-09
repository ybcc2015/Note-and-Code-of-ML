# coding: utf-8
import tensorflow as tf
from helper import get_iris_data


def PLA_tf(X, Y, lr=0.01, max_epochs=1000, log_step=50):
    """
    tensorflow版感知机
    :param X: 2-D array, shape(num_samples, num_features), 包含了所有训练数据的feature
    :param Y: 1-D array, shape(num_samples,), 包含了所有训练样本的label
    :param lr: 学习率
    :param max_epochs: 最大迭代次数
    :param log_step: 打印日志间隔
    :return: None
    """
    # tf计算图的输入
    X_input = tf.placeholder(tf.float32, shape=(None, 2), name='X')
    Y_input = tf.placeholder(tf.float32, shape=(None, 1), name='Y')

    # 初始化参数
    W = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32, name='weight')
    b = tf.Variable(0.0, dtype=tf.float32, name='bias')

    # y * (wx + b)
    pred = tf.squeeze(tf.multiply(Y_input, tf.matmul(X_input, W) + b))

    # loss
    loss = -tf.reduce_sum(pred)

    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 训练
    for epoch in range(max_epochs):
        wrong_samples = []  # 用来存放误分类样本loss
        for x, y in zip(X, Y):
            feed_dict = {X_input: x.reshape(-1, 2), Y_input: y.reshape(-1, 1)}
            loss = sess.run(pred, feed_dict=feed_dict)
            if loss < 0:
                sess.run(optimizer, feed_dict=feed_dict)
                wrong_samples.append(loss)

        if not wrong_samples:
            print('epoch: {}\tloss: 0.0'.format(epoch))
            break

        if epoch % log_step == 0:
            total_loss = sess.run(-tf.reduce_sum(wrong_samples))
            print('epoch: {}\tloss: {:.4f}'.format(epoch, total_loss))
    print('training finished!')


if __name__ == '__main__':
    X, Y = get_iris_data()
    PLA_tf(X, Y)
