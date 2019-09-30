#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector


# In[2]:


#载入MNIST数据集,使用one_hot编码
mnist = input_data.read_data_sets("mnist", one_hot=True)
#运行次数
max_steps = 1001
#图片数量
image_num = 3000
#文件路径
DIR = "F:/program/jupyter/"

#定义会话
sess = tf.Session()

#载入图片，tf.stack方法用于组合，打包数据，这里打包3000张图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')

#定义统计函数
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev) #标准差
        tf.summary.scalar('max', tf.reduce_max(var)) #最大值
        tf.summary.scalar('min', tf.reduce_min(var)) #最小值
        tf.summary.histogram('histogram', var) #直方图

#输入
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

#显示图片
with tf.name_scope('input_reshape'):
    image_shape_input = tf.reshape(x, [-1,28,28,1]) #图片尺寸28*28，黑白图片
    tf.summary.image('input', image_shape_input,10) #放10张图片

#创建神经网路
with tf.name_scope('layer'):
    #权值
    W = tf.Variable(tf.zeros([784,10]), name='w')
    variable_summaries(W)
    #偏置值
    b = tf.Variable(tf.zeros([10]), name='b')
    variable_summaries(b)
    #结果
    prediction = tf.matmul(x,W)+b
    
#使用交叉熵
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
    
with tf.name_scope('train'):
    learning_rate = 0.5
    #使用梯度下降法优化
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess.run(tf.global_variables_initializer())

with tf.name_scope('cal_accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

#产生metadata文件
if tf.gfile.Exists(DIR + 'Projector/projector/metadata.tsv'):
    tf.gfile.DeleteRecursively(DIR + 'Projector/projector/metadata.tsv') #有则删除
with open(DIR + 'Projector/projector/metadata.tsv', 'w') as f:
    labels = sess.run(tf.argmax(mnist.test.labels[:],1)) #获取测试集标签最大值位置
    for i in range(image_num):
        f.write(str(labels[i]) + '\n')  #写入标签位置

#合并summary
merged = tf.summary.merge_all()

projector_writer = tf.summary.FileWriter(DIR + 'Projector/projector',sess.graph) #用于写入图结构
saver = tf.train.Saver() #用于保存网络模型
config = projector.ProjectorConfig() #定义配置文件
embed = config.embeddings.add() #配置
embed.tensor_name = embedding.name
embed.metadata_path = DIR + 'Projector/projector/metadata.tsv'
embed.sprite.image_path = DIR + 'Projector/data/mnist_10k_sprite.png'
embed.sprite.single_image_dim.extend([28,28]) #切分28*28
projector.visualize_embeddings(projector_writer,config) #记录

#训练，1001*100次
for i in range(max_steps):
    #每批次100个样本
    batch_xs,batch_ys = mnist.train.next_batch(100)
    run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys}, options=run_options, run_metadata=run_metadata)
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i) #记录
    
    if i%100 == 0:
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("iter " + str(i) + ",Testing Accuracy " + str(acc))

saver.save(sess, DIR + 'Projector/projector/a_model.ckpt', global_step=max_steps) #保存模型
projector_writer.close()
sess.close()


# In[ ]:


#

