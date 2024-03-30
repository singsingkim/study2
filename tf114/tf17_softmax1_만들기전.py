import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
tf.set_random_seed(777)

# 1 데이터
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,6,7]]

y_data = [[0,0,1],  # 2
          [0,0,1],
          [0,0,1],
          [0,1,0],  # 1
          [0,1,0],
          [0,1,0],
          [1,0,0],  # 0
          [1,0,0]]

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
w = tf.compat.v1.Variable(tf.compat.v1.random_normal([4, 3]), name='weight')
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1, 3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])


# 2 모델
hypothesis = tf.compat.v1.matmul(x, w) + b

# 3-1 컴파일
loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y)) # mse
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 1001

for step in range(epoch):
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    
    if step % 20 == 0 :
        print(step, 'loss : ', cost_val)
        
print(type(w_v))    # <class 'numpy.ndarray'>
                    # 텐서플로 에서는 넘파이로 생성
                    
# x_tset = tf.compat.v1.placeholder(tf.float32, shape=([None, 4]))

y_pred = tf.matmul(x, w_v) + b_v

y_predict = sess.run(y_pred, feed_dict={x:x_data})
print(y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_data, y_predict)
mse = mean_squared_error(y_data, y_predict)

print('r2 : ', r2)      
print('mse : ', mse)    

sess.close()

# 위의 결과는 회귀