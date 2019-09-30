#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[7]:


#使用numpy生成100个随机点
x_data = np.random.rand(100)
#优化目标
y_data = x_data * 1.1 + 0.5

#构造线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

#二次代价函数（损失函数），使用平均方差（最小均方误差）
loss = tf.reduce_mean(tf.square(y_data-y))
#定义使用梯度下降法优化器来优化（学习率为0.2）
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#训练会使b和k的值接近上面y_data的b和k
#定义会话
with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        #每次迭代run（train），
        sess.run(train)
        if step%20 == 0:
            print(step, ": ", sess.run([k,b]))


# In[ ]:




