#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


#载入MNIST数据集,使用one_hot编码
mnist = input_data.read_data_sets("mnist", one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共多少个批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值,参数权值形状
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) #生成一个截断的正态分布，标准差0.1
    return tf.Variable(initial)

#初始化偏置值
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    #参数x，一个输入张量（tensor），形状为4维【批次（会置换成上面定义的批次大小），长，宽，通道数（深度）】
    #参数W，filter的张量，【长，宽，输入通道数（输入深度），输出通道数（输出深度）】
    #strides[0]=strides[3]=1，strides[1]代表x方向的步长，strides[2]代表y方向上的步长
    #padding，选择填充方式，有'SAME'和'VALID'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    #使用最大池化方法
    #ksize和strides类似,[1][2]分别代表窗口长和宽
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784]) #28x28=784
y = tf.placeholder(tf.float32,[None,10])

#改变x的格式转为四维向量【batch, in_height, in_weight, in_channels】
#复原图片样子，把784展开，-1会替换成100，深度1一定与上层深度一样，这里1表示图片为黑白，若是彩色则有rgb3层，下同
x_image = tf.reshape(x,[-1,28,28,1])

#开始前向传播

#初始化第一个卷积层的权值和偏置值
W_conv1 = weight_variable([5,5,1,32]) #使用5x5采样窗口，共32个filter
b_conv1 = bias_variable([32])

#对x_image进行卷积计算然后池化，激活函数使用Relu函数
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 =  max_pool_2x2(h_conv1) #此时图片由28x28变为14x14，计算公式如下

#第二层
W_conv2 = weight_variable([5,5,32,64]) #使用5x5采样窗口，共64个filter
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) #此时图片由14x14变为7x7,计算公式如下

#输入拆分为h和w（下面有点难理解）
#h_output=((h_in-filter_in+2*padding)/s)+1
#（h_in代表h的输入，filter_in代表filter的输入h，padding代表填充了多少层0，s代表filter的步进）
#所以经过上面操作后得到64张7*7的平面

#初始化第一个全连接层
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

#把池化层2的输出转为1维向量
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#第一个全连接层的输出
h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#设置Dropout
keep_prob = tf.placeholder(tf.float32)
h_fcl_drop = tf.nn.dropout(h_fcl,keep_prob)

#第二层全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
#计算输出
prediction = tf.matmul(h_fcl_drop,W_fc2) + b_fc2

#反向传播

#损失函数/交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用Adam优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#存放结果的布尔列表
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
            
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:




