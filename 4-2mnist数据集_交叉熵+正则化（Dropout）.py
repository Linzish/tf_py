#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#定义dropout率，取值在0-1，代表dropout百分比，1.0表示所有神经元工作
keep_prob = tf.placeholder(tf.float32)

#创建神经网络
#权值&偏置值,一般使用正态分布初始化（标准差0.1）,激活函数使用tanh，设置dropout
W1 = tf.Variable(tf.truncated_normal([784,2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)
#添加隐藏层
W2 = tf.Variable(tf.truncated_normal([2000,2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000,1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3, keep_prob)

#输出层（使用softmax函数，softmax函数会把输入值转化为概率值）
W4 = tf.Variable(tf.truncated_normal([1000,10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.matmul(L3_drop,W4)+b4

#交叉熵损失函数
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
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
        #获取准确率，传入测试集的图片和标签
        test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels, keep_prob:1.0})
#         train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images,y:mnist.train.labels, keep_prob:1.0})
#         print("iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + " ,Training Accuracy " + str(train_step))
        print("iter " + str(epoch) + ",Testing Accuracy " + str(test_acc))


# In[ ]:




