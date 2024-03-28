# 회귀
# 7. load_diabetes
# 8. california
# 9. dacon_ddarung
# 10. kaggle_bike

import tensorflow as tf
tf.compat.v1.set_random_seed(777)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
#1. 데이터

path = "c:/_data/kaggle/bike//"

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

######### 결측치 처리 1. 제거 #########
train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
test_csv = test_csv.fillna(0)                       # 널값에 0 을 넣은거

######### x 와 y 를 분리 #########
x = train_csv.drop(['count','casual','registered'], axis = 1)     # count를 삭제하는데 count가 열이면 액시스 1, 행이면 0
y = train_csv['count']
print(x.shape, y.shape) #(10886, 8) (10886,)



x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=777, shuffle=True,
    )

xp = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
yp = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

w= tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1]), dtype=tf.float32, name='weight')
b= tf.compat.v1.Variable(tf.compat.v1.zeros([1]), dtype=tf.float32, name='bias')


#2 모델
hypothesis = tf.compat.v1.matmul(xp, w) + b


# 3-1 컴파일
# loss = tf.reduce_mean(tf.square(hypothesis - y)) # mse
# 3-1 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - tf.cast(y, dtype=tf.float32))) # mse



optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)


# 3-2 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 101
for step in range(epochs):
    # 맹그러
    cost_val, _, w_v, b_v = sess.run([loss, train, w, b], feed_dict={xp:x_train, yp:y_train})

    if step % 20 == 0:
        print(step, 'loss : ', cost_val)


x_pred = tf.compat.v1.placeholder(tf.float32, shape=[None, 8])
y_pred = tf.matmul(x_pred, w_v) + b_v
y_pred = sess.run(y_pred, feed_dict={x_pred:x_test})

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print('y_pred : ', y_pred)
print('r2 : ', r2)
print('mse : ', mse)

sess.close()        

# y_pred :  [[ -73.70828 ]
#  [-213.72739 ]
#  [ -72.802574]
#  ...
#  [-276.25592 ]
#  [ -45.280636]
#  [-190.59607 ]]
# r2 :  -60213.81381228538
# mse :  77619.85793328521