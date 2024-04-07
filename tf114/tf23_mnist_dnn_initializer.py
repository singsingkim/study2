import tensorflow as tf
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(x_train.shape, x_test.shape)  # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)
print(y_test)


x_train = x_train.reshape(60000, 28*28).astype('float32')/255.
x_test = x_test.reshape(10000, 28*28).astype('float32')/255.

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)
print(y_train.shape, y_test.shape)  # (60000, 10) (10000, 10)

keep_prob = tf.compat.v1.placeholder(tf.float32)

# [실습] 맹그러바

x = tf.compat.v1.placeholder(tf.float32,shape=[None, 784])
y = tf.compat.v1.placeholder(tf.float32,shape=[None, 10])

w1 = tf.compat.v1.get_variable('weight1', shape=[784, 128],
                               initializer=tf.contrib.layers.xavier_initializer()) # 가중치 초기화 // 
                                                                                    # 자동으로 random_noramalize 적용
                                                                                    # 가중치 초기화는 첫 에포에만 적용된다(if문 사용으로)
b1 = tf.compat.v1.Variable(tf.zeros([128], name='bias1'))
layer1 = tf.compat.v1.matmul(x, w1) + b1
layer1 = tf.compat.v1.nn.dropout(layer1, rate=0.3)

w2 = tf.compat.v1.get_variable('weight2',shape=[128, 64],
                                initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.zeros([64], name='bias2'))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    
layer2 = tf.compat.v1.nn.relu(layer2)
layer2 = tf.compat.v1.nn.dropout(layer2, rate=0.3)

w3 = tf.compat.v1.Variable(tf.random_normal([64, 32], name='weight3')) 
b3 = tf.compat.v1.Variable(tf.zeros([32], name='bias3'))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    
layer3 = tf.compat.v1.nn.relu(layer3)
layer3 = tf.compat.v1.nn.dropout(layer3, rate=0.3)

w4 = tf.compat.v1.Variable(tf.random_normal([32,10], name='weight4')) 
b4 = tf.compat.v1.Variable(tf.zeros([10], name='bias4'))
layer4 = tf.compat.v1.matmul(layer3,w4) + b4

hypothesis = tf.compat.v1.nn.softmax(layer4)

# 3-1 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.nn.log_softmax(hypothesis), axis=1))
# loss = tf.comopat.v1.losses.softmax_cross_entropy(y, hypothesis)

train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
  
           
# 3-2 훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())   # 변수를 메모리에 올린다.

epochs = 101
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w4, b4], feed_dict={x:x_train, y:y_train})
    
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)
    

# 4 평가, 예측
print('===========================================================')
y_predict = sess.run(hypothesis, feed_dict={x:x_test})
print(y_predict, y_predict.shape)   # (10000, 10)

y_predict_arg = sess.run(tf.arg_max(y_predict, 1))
print(y_predict_arg[0], y_predict_arg.shape)    # 7 (10000,)

import numpy as np
y_test = np.argmax(y_test, 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict_arg, y_test)
print('acc : ', acc)

# acc :  0.7991