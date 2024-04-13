import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1 데이터
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.


#2 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 28, 28, 1])  # 4차원
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])  


# layer1
w1 = tf.compat.v1.get_variable('w1', shape=[2, 2, 1, 128], initializer=tf.contrib.layers.xavier_initializer())
                                            # 커널사이즈, 컬러(채널), 필터(아웃풋)

b1 = tf.compat.v1.Variable(tf.zeros([128]), name='b1')

l1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='VALID') # 4차원 맞추기 위한 형식 / 세칸 뛰우고 싶으면 : [1,3,3,1]
# model.add(Conv2d(64, kernel_size=(2,2), input_shape=(28,28,1), stride=(1,1)))
l1 += b1    # l1 = l1 + b1
l1 = tf.nn.relu(l1)
l1 = tf.nn.dropout(l1, keep_prob=0.7)
# l1 = tf.nn.dropout(l1, rate=0.3)    # model.add(Dropout(0.3))
l1_maxpool = tf.nn.max_pool2d(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(l1)           # Tensor("Relu:0", shape=(?, 27, 27, 128), dtype=float32)
print(l1_maxpool)   # Tensor("MaxPool2d:0", shape=(?, 13, 13, 128), dtype=float32)   # 나머지 날린다 = valid, 나머지 살린다 = same
   

# layer2
w2 = tf.compat.v1.get_variable('w2', shape=[3, 3, 128, 64], initializer=tf.contrib.layers.xavier_initializer())
                                            # 커널사이즈, 컬러(채널), 필터(아웃풋)

b2 = tf.compat.v1.Variable(tf.zeros([64]), name='b2')

l2 = tf.nn.conv2d(l1_maxpool, w2, strides=[1,1,1,1], padding='SAME') 

l2 += b2
l2 = tf.nn.selu(l2)
l2 = tf.nn.dropout(l2, keep_prob=0.9)
# l1 = tf.nn.dropout(l1, rate=0.1)    # model.add(Dropout(0.1))
l2_maxpool = tf.nn.max_pool2d(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(l2)           # 
print(l2_maxpool)   # 



# layer2
w3 = tf.compat.v1.get_variable('w3', shape=[3, 3, 64, 32], initializer=tf.contrib.layers.xavier_initializer())
                                            # 커널사이즈, 컬러(채널), 필터(아웃풋)

b3 = tf.compat.v1.Variable(tf.zeros([32]), name='b3')

l3 = tf.nn.conv2d(l2_maxpool, w3, strides=[1,1,1,1], padding='SAME') 

l3 += b3
l3 = tf.nn.selu(l3)
l3_maxpool = tf.nn.max_pool2d(l3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
print(l3)           # 
print(l3_maxpool)   # 


# Flatten
l_flat = tf.reshape(l3, [-1, 6*6*32])
print('플랫 : ', l_flat)    # 플랫 :  Tensor("Reshape:0", shape=(?, 1152), dtype=float32)


# layer4 dnn
w4 = tf.compat.v1.get_variable('w4', shape=[6*6*32, 100])
b4 = tf.compat.v1.Variable(tf.zeros([100]), name='b4')
l4 = tf.nn.relu(tf.matmul(l_flat, w4) + b4)
l4 = tf.nn.dropout(l4, rate=0.3)


# layer5 dnn
w5 = tf.compat.v1.get_variable('w5', shape=[100, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.compat.v1.Variable(tf.zeros([10]), name='b5')
l5 = tf.compat.v1.matmul(l4, w5) + b5
hypothesis = tf.compat.v1.nn.softmax(l5)

