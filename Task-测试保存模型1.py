#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[5]:


mnist = input_data.read_data_sets("mnist", one_hot=True)

#加载模型
saver = tf.train.import_meta_graph('./models/test/test.ckpt.meta')


graph = tf.get_default_graph()
#x = graph.get_tensor_by_name("x-input:0")
#y = graph.get_tensor_by_name("y-input:0")
#keep_prob_ = graph.get_tensor_by_name("k-prob:0")



#y_ = graph.get_tensor_by_name("y-out:0")

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    saver.restore(sess, tf.train.latest_checkpoint('./models/test/'))
    
    x = graph.get_tensor_by_name("x-input_2:0")
    y = graph.get_tensor_by_name("y-input_2:0")
    accuracy = sess.graph.get_tensor_by_name("accuracy:0")
    
    
    acc = sess.run(accuracy, feed_dict={x:mnist.test.images,y:mnist.test.labels})
    print("iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:




