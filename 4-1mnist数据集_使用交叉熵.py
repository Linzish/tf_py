#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


#载入MNIST数据集,使用one_hot编码
mnist = input_data.read_data_sets("mnist", one_hot=True)

#定义batch大小，一次性100张图
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder，数据集图片为28*28(=784)，把图片转化为一维向量
x = tf.placeholder(tf.float32, [None, 784])
#一共十个标签
y = tf.placeholder(tf.float32, [None, 10])

#创建神经网络,输入784个节点，输出10个节点
#权值
W = tf.Variable(tf.zeros([784,10]))
#偏置值
b = tf.Variable(tf.zeros([10]))
#激活函数（使用softmax函数，softmax函数会把输入值转化为概率值）
# prediction = tf.nn.softmax(tf.matmul(x,W)+b)
#下面交叉熵包含softmax操作
prediction = tf.matmul(x,W)+b

#二次代价函数/损失函数
# loss = tf.reduce_mean(tf.square(y-prediction))
#使用交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
#学习率
learning_rate = 0.2
#使用梯度下降法优化
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

#定义准确率
#tf.equal函数对比两个输入是否一样，返回布尔类型，
#所以correct_prediction为一个true和false的集合（布尔型列表）
#tf.argmax求行中的最大值索引位置（第二参数为0时按列算，为1按行算）
#！！注意y使用one_hot编码
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
#求准确率
#把correct_prediction转换为浮点类型再求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#会话
with tf.Session() as sess:
    sess.run(init)
    #迭代21个周期（21*100）（所有图片训练21次）
    for epoch in range(21):
        for batch in range(n_batch):
            #传入100张图片，数据保存在batch_xs和batch_ys中
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
        #获取准确率，传入测试集的图片和标签
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:


iter 0,Testing Accuracy 0.8298
iter 1,Testing Accuracy 0.8706
iter 2,Testing Accuracy 0.882
iter 3,Testing Accuracy 0.8873
iter 4,Testing Accuracy 0.8942
iter 5,Testing Accuracy 0.8974
iter 6,Testing Accuracy 0.8988
iter 7,Testing Accuracy 0.902
iter 8,Testing Accuracy 0.9035
iter 9,Testing Accuracy 0.9051
iter 10,Testing Accuracy 0.9062
iter 11,Testing Accuracy 0.907
iter 12,Testing Accuracy 0.9079
iter 13,Testing Accuracy 0.9093
iter 14,Testing Accuracy 0.9093
iter 15,Testing Accuracy 0.9107
iter 16,Testing Accuracy 0.9122
iter 17,Testing Accuracy 0.9118
iter 18,Testing Accuracy 0.9126
iter 19,Testing Accuracy 0.9138
iter 20,Testing Accuracy 0.914

