#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
#用于绘图
import matplotlib.pyplot as plt


# In[5]:


#定义样本
#使用numpy生成200个随机点
#范围：-0.5~0.5，后面加一个维度（200行一列）
x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
#生成一些随机干扰项，与x_data一样维度
noise = np.random.normal(0, 0.02, x_data.shape)
#得到图像分布形状大致为抛物线
y_data = np.square(x_data) + noise

#定义两个placeholder，维度中的None表示任意，但只有一列（跟上面样本对应）
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义神经网络（输入层为一个节点，中间层10个节点，输出层1个节点）
#定义权重和偏置值（1转10）
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
#输入，矩阵相乘+偏置值，然后使用激活函数非线性化（tanh函数），得到中间层输出L1
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定于输出层
#10转1，输出是一个数所以偏置值是一个标量
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
#prediction为预测结果
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#定义学习率
learning_rate = 0.1
#定义使用梯度下降法优化器来优化（学习率为0.2）
#最小化代价函数
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

#会话
with tf.Session() as sess:
    sess.run(init)
    #训练2000次
    for _ in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
        
    #获得预测值，测试只需要传入x
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    #画图看效果
    plt.figure()
    #先使用散点图的方式打印样本点
    plt.scatter(x_data, y_data)
    #画出预测值分布，'r-'意思是使用红色实线，lw为线宽度
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()


# In[ ]:




