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
# [[1,2,1],
#  [1,2,1],
#  [1,2,1]]
b = tf.compat.v1.Variable(tf.compat.v1.zeros([1, 3]), name='bias')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3])


# 2 모델
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(x, w) + b)

# 3-1 컴파일
# loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y)) # mse
loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ Categorical_Crossentropy ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(loss)


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 100001
for step in range(epoch):
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b], 
                                     feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss : ', cost_val)
    
print(w_v)  # 4 * 3 = 12개 나옴
# [[-0.96051157 -0.40883178 -0.4266268 ]
#  [ 0.18936276 -0.0649445   0.7510862 ]
#  [ 0.8896435   0.12780887  0.26152438]
#  [-0.13477631  0.6172665  -0.36967286]]
print(b_v)
# [[ 0.0470526   0.05709783 -0.10415049]]


# 4 평가, 예측
y_predict = sess.run(hypothesis, feed_dict={x:x_data})

print(y_predict)    # (8, 3)
# [[5.5693682e-02 6.6997521e-02 8.7730879e-01]
#  [2.2315778e-01 3.3748576e-01 4.3935651e-01]
#  [5.2746832e-02 8.0712622e-01 1.4012702e-01]
#  [5.8047906e-02 8.5147023e-01 9.0481848e-02]
#  [6.5246437e-05 2.7720824e-05 9.9990702e-01]
#  [7.9949416e-02 3.0463243e-01 6.1541814e-01]
#  [3.7879040e-04 2.2834478e-04 9.9939287e-01]
#  [7.0485810e-05 7.4901283e-05 9.9985456e-01]]

y_predict = sess.run(tf.arg_max(y_predict, 1))
print(y_predict)
# [2 2 1 1 2 2 2 2]

# y_predict = np.argmax(y_predict, 1)
print(y_predict)

y_data = np.argmax(y_data, 1)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)
print('acc : ', acc)

sess.close()

# 에포 100000
# acc :  0.625