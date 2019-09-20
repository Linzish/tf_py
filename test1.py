#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
#定义激活函数（sigmoid函数）
#一种写法
#if(derivative == True):
#        return x * (1.0 - x)
#    else:
#        return 1.0 / (1.0 + np.exp(-x))
def sigmoid(x, derivative = False):
    sigmoid = 1.0/(1.0+np.exp(-x))
    if (derivative==True):
        return sigmoid * (1-sigmoid)
    return sigmoid


# In[48]:


#定义输入数据x
x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
    [0,0,1]
])
print(x.shape)


# In[49]:


#定义目标数据y
y = np.array([
    [0],
    [1],
    [1],
    [0],
    [0]
])
print(y.shape)


# In[50]:


#定义随机种子
np.random.seed(1)


# In[51]:


#定义一个3层神经万诺
#需要两个权重W0，W1，其值随机定义
#W0为3*4矩阵，因为输入数x有3列（参考线代知识）
#w0 = np.random.random((3,4))
#W1为 4*1矩阵
#w1 = np.random.random((4,1))

#添加限定，使得W0、W1落在区间【-1，1】上
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1

print(w0)


# In[52]:


#神经网络迭代计算
for j in range(60000):
    # !!定义神经网络的前向传播过程（使用sigmoid作为激活函数）
    #定义输入层l0
    l0 = x
    #定义神经网络第一层，一层操作为权值计算（+偏置值）+非线性化，即sigmoid（l0*W0(+b)）
    #np.dot做矩阵乘法运算
    l1 = sigmoid(np.dot(l0, w0))
    #定义神经网络第二层
    l2 = sigmoid(np.dot(l1, w1))
    
    # !!定义神经网络反向传播过程
    #计算神经网络计算误差（定义损失函数/优化函数）,使用函数f(x)=(y-y~)**2 / 2（均方误差）（？求导使用）
    #l2_error为真实值和预测值之间的差异值
    l2_error = y - l2
    #测试效果，打印当前误差值
    #np.mean用于求均值,np.abs用于求绝对值
    if (j%10000) == 0:
        print('Error: ' + str(np.mean(np.abs(l2_error))))
    #反向传播过程：
    #l2_delta含义：l2层错了多少,贡献多少错误。把l2_error看成是错误权重，权重越大就越需要修改，乘上sigmoid函数的导数值（参考梯度下降法）
    #运算为普通乘法,获得对应元素相乘的值
    l2_delta = l2_error * sigmoid(l2, derivative=True)
    #运算为矩阵乘法乘上w1的转置矩阵
    #开始梯度传播
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid(l1, derivative=True)
    #更新权重W0，W1
    #l1的转置乘上上一层传下来的误差
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)


# In[ ]:





# In[ ]:




