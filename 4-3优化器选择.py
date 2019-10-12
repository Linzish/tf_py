#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[7]:


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

#！！这里修改优化器

#使用梯度下降法优化

# GradientDescentOptimizer优化器，标准梯度下降法（GD）
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# MomentumOptimizer优化器，动量优化算法，动量优化算法在梯度下降法（SGD）的基础上进行改变，具有加速梯度下降的作用。
# 当前权值的改变会受到上一次权值改变的影响。
# momentum（动量），表示要在多大程度上保留原来的更新方向，这个值在0-1之间。
# 在训练开始时，由于梯度可能会很大，所以初始值一般选为0.5；当梯度不那么大时，改为0.9。
# 特点：前后梯度方向一致时,能够加速学习；前后梯度方向不一致时,能够抑制震荡
#train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.5).minimize(loss)

# AdagradOptimizer优化器，基于SGD
# 核心思想是对比比较常见得数据给予它比较小的学习率调整参数，对于比较罕见的数据给予它比较大的学习率去调整。
# 适合用于数据稀疏的数据集。优势在于不需要人为调节学习率，缺点是随着迭代次数增多，学习率会越来越小，趋向于0。
#train_step = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)

# RMSPropOptimizer优化器，使用的是指数加权平均。用前t次梯度平方的平均值加上当前梯度的平方的和的开平方作为学习率的分母。
# 自适应的学习率调参方法，目的是为了减少人工调参的次数（只是减少次数，还是需要人工设定学习率的）
# 修改了AdaGrad的梯度累积为指数加权的移动平均，使在非凸下效果更好。
# decay代表衰减系数
#train_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.8, momentum=0.1).minimize(loss)

# AdadeltaOptimizer优化器，实现一个自适应的、单调递减的学习率，它使用两个初始化参数 learning_rate 和衰减因子 rho。
# AdaGrad与RMSProp都需要指定全局学习率，AdaDelta结合两种算法每次参数的更新步长。
# 在训练的前中期，表现效果较好，加速效果可以，训练速度更快。在后期，模型会反复地在局部最小值附近抖动。
#train_step = tf.train.AdadeltaOptimizer(rho=0.95).minimize(loss)

# AdamOptimizer优化器，自适应矩估计。本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
# 就像Adadelta和RMSProp一样，Adam会存储之前衰减的平方梯度，同时也会保存之前衰减的梯度。经过一些处理后再使用类似Adadelta和RMSProp的方式更新参数。
# Adam的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
# 相比于，RMSProp缺少修正因子导致二阶矩估计在训练初期有较高的偏置，Adam包括偏置修正，从原始点初始化的一阶矩（动量项）和（非中心的）二阶矩估计。
# 常用
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 不同优化器比较
# 下降速度上，三个自适应学习优化器  AdaGrad,RMSProp与AdaDelta的下降速度明显快于SGD，而Adagrad与
# RMSProp速度相差不大快于AdaDelta。两个动量优化器Momentum ,NAG初期下降较慢，后期逐渐提速，NAG后期超过Adagrad与RMSProt。
# 在有鞍点的情况下，自适应学习率优化器没有进入，Momentum与NAG进入后离开并迅速下降。而SGD进入未逃离鞍点。
# 速度：快->慢： Momenum ，NAG -> AdaGrad,AdaDelta,RMSProp ->SGD
# 收敛： 动量优化器有走岔路，三个自适应优化器中Adagrad初期走了岔路，
# 但后期调整，与另外两个相比，走的路要长，但在快接近目标时，RMSProp抖动明显。SGD走的过程最短，而且方向比较正确。

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




