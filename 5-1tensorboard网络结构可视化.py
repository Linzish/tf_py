#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[11]:


#载入MNIST数据集,使用one_hot编码
mnist = input_data.read_data_sets("mnist", one_hot=True)

#定义batch大小，一次性100张图
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#主要是添加命名空间

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    W = tf.Variable(tf.zeros([784,10]), name='w')
    b = tf.Variable(tf.zeros([10]), name='b')
    prediction = tf.matmul(x,W)+b

#二次代价函数/损失函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

with tf.name_scope('train'):
    learning_rate = 0.2
    #使用梯度下降法优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


init = tf.global_variables_initializer()

#定义准确率
#tf.equal函数对比两个输入是否一样，返回布尔类型，
#所以correct_prediction为一个true和false的集合（布尔型列表）
#tf.argmax求行中的最大值索引位置（第二参数为0时按列算，为1按行算）
#！！注意y使用one_hot编码
with tf.name_scope('cal_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#会话
with tf.Session() as sess:
    sess.run(init)
    #生成tensorborad文件
    writer = tf.summary.FileWriter('logs/5-1/', sess.graph)
    #迭代21个周期（21*100）（所有图片训练21次）
    for epoch in range(1):
        for batch in range(n_batch):
            #传入100张图片，数据保存在batch_xs和batch_ys中
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        #获取准确率，传入测试集的图片和标签
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:




