#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


def imageprepare(): 
    im = Image.open('F:/learning/tensorflow/test-1.png') #读取的图片所在路径，注意是28*28像素
    plt.imshow(im)  #显示需要识别的图片
    plt.show()
    im = im.convert('L')
    tv = list(im.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva

result=imageprepare()


mnist = input_data.read_data_sets("mnist", one_hot=True)
#定义batch大小，一次性100张图
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder，数据集图片为28*28(=784)，把图片转化为一维向量
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
#一共十个标签
y = tf.placeholder(tf.float32, [None, 10], name='y-input')

#创建神经网络,输入784个节点，输出10个节点
#权值
W = tf.Variable(tf.zeros([784,10]))
#偏置值
b = tf.Variable(tf.zeros([10]))
#激活函数（使用softmax函数，softmax函数会把输入值转化为概率值）
# prediction = tf.nn.softmax(tf.matmul(x,W)+b)
#下面交叉熵包含softmax操作
with tf.name_scope('prediction'):
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
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

saver = tf.train.Saver()

#会话
with tf.Session() as sess:
    sess.run(init)
    # 导入模型
    saver.restore(sess, tf.train.latest_checkpoint('./models/test/'))
    # 测试
    pre=tf.argmax(prediction,1)
    predint=pre.eval(feed_dict={x: [result]}, session=sess)
    
    print('识别结果:')
    print(predint[0])


# In[ ]:




