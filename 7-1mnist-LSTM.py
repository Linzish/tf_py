#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


#载入MNIST数据集,使用one_hot编码
mnist = input_data.read_data_sets("mnist", one_hot=True)

#输入图片为28*28
n_inputs = 28 #输入一行，一行有28个数据
max_time = 28 #一共28行，一共输入28次28*28
lstm_size = 100 #隐藏层基础单元
n_classes = 10 #10个分类
batch_size = 50 #每批次50个样本
n_batch = mnist.train.num_examples // batch_size #计算一共付给几个批次

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#初始化权值，偏置值
weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes], stddev=0.1)) #一共100个单元，10个分类（100*10矩阵）
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

#定义RNN网络
def RNN(X,weights,biases):
    #输入数据转换  [batch_size,max_time,n_inputs]
    inputs = tf.reshape(X,[-1,max_time,n_inputs]) # ->[50,28,28]
    #定义LSTM的基本单元cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    
    # final_state[0]为cell state（？候选值）
    # final_state[1]为hidden_state（？已更新输入）
    #-----------------------------------
    # 当输入的cell为BasicLSTMCell时，state的形状为[2，batch_size, cell.output_size ]，其中2也对应着LSTM中的cell state和hidden state
    # 如果cell为LSTM，那 state是个tuple，分别代表 和 ，其中 与outputs中的对应的最后一个时刻的输出相等，
    # 假设state形状为[ 2，batch_size, cell.output_size ]，outputs形状为 [ batch_size, max_time, cell.output_size ]，
    # 那么state[ 1, batch_size, : ] == outputs[ batch_size, -1, : ]；
    # 如果cell为GRU，那么同理，state其实就是 ，state ==outputs[ -1 ]
    #-----------------------------------
    # outputs是每个step的输出，它的结构是[batch_size，step，n_neurons]
    # final_states是每一层的最后那个step（记忆层c）的输出
    # 这个输出有两个信息，一个是h表示短期记忆信息，一个是c表示长期记忆信息。维度都是[batch_size，n_neurons]
    # states的最后一个LSTMStateTuple中的h就是outputs的最后一个step的输出
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32) #output[50,28,100] state[50,100]
#     result = tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    result = tf.matmul(final_state[1],weights)+biases
    return result

#计算RNN的返回结果
prediction = RNN(x, weights, biases)
#损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用Adam优化器
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#存放结果的布尔列表
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()

#会话
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:




