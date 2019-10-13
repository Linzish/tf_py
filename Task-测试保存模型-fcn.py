#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# In[2]:


def imageprepare(): 
    im = Image.open('F:/learning/tensorflow/test-7.png') #读取的图片所在路径，注意是28*28像素
    plt.imshow(im)  #显示需要识别的图片
    plt.show()
    im = im.convert('L')
    tv = list(im.getdata()) 
    tva = [(255-x)*1.0/255.0 for x in tv] 
    return tva

result=imageprepare()


#载入MNIST数据集,使用one_hot编码
mnist = input_data.read_data_sets("mnist", one_hot=True)

#定义batch大小，一次性100张图
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder，数据集图片为28*28(=784)，把图片转化为一维向量
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
#一共十个标签
y = tf.placeholder(tf.float32, [None, 10], name='y-input')
#定义dropout率，取值在0-1，代表dropout百分比，1.0表示所有神经元工作
keep_prob = tf.placeholder(tf.float32, name='k-prob')
#定义可修改学习率
lr = tf.Variable(0.001, dtype=tf.float32, name='learning-rate')

#创建神经网络
#权值&偏置值,一般使用正态分布初始化（标准差0.1）,激活函数使用tanh，设置dropout
W1 = tf.Variable(tf.truncated_normal([784,500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1, keep_prob)
#添加隐藏层
W2 = tf.Variable(tf.truncated_normal([500,300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.matmul(L2_drop,W3)+b3

#交叉熵损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='loss')
#使用梯度下降法优化
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
#Adam优化器
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

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

#保存模型的路劲
#ckpt_file_path = "models/fcn/"
# path = os.path.dirname(os.path.abspath(ckpt_file_path))
# if os.path.isdir(path) is False:    
#     os.makedirs(path)
#用于保存模型
saver = tf.train.Saver()

#会话
with tf.Session() as sess:
    sess.run(init)
    # 导入模型
    saver.restore(sess, tf.train.latest_checkpoint('./models/fcn/'))
    # 测试
    pre=tf.argmax(prediction,1)
    predint=pre.eval(feed_dict={x: [result], keep_prob:1.0}, session=sess)
    
    print('识别结果:')
    print(predint[0])


# In[ ]:




