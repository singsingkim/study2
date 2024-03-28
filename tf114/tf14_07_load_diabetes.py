# 회귀
# 7. load_diabetes
# 8. california
# 9. dacon_ddarung
# 10. kaggle_bike

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1 데이터
x, y = load_diabetes(return_X_y=True)
print(x.shape, y.shape) # (442, 10) (442,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=777, shuffle=True
    )

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

w= tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 1]), dtype=tf.float32, name='weight')
b= tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32, name='bias')


#2 모델
hypothesis = tf.compat.v1.matmul(xp, w) + b


# 3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 10001
for step in range(epochs):
    # 맹그러
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b], feed_dict={xp:x_train, yp:y_train})

    if step % 20 == 0:
        print(step, 'loss : ', cost_val)


x_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
y_pred = tf.matmul(x_pred, w_v) + b_v
y_pred = sess.run(y_pred, feed_dict={x_pred:x_test})

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('y_pred : ', y_pred)
print('r2 : ', r2)
print('mse : ', mse)

sess.close()        


# y_pred :  [[ 9.963267 ]
#  [ 9.943188 ]
#  [11.20813  ]
#  [11.081884 ]
#  [11.122364 ]
#  [ 9.312362 ]
#  [11.992066 ]
#  [11.215096 ]]
# r2 :  -4.05249012888968
# mse :  27685.47718185664