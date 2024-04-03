# 분류
# 1. 캔서(이진)
# 2. digits
# 3. fetch_covtype
# 4. dacon_wine
# 5. dacon_dechul
# 6. kaggle_biman

# 회귀
# 7. load_diabets
# 8. california
# 9. dacon_ddarung
# 10. kaggle_bike

import tensorflow as tf
tf.compat.v1.set_random_seed(123)

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
x,y = fetch_california_housing(return_X_y=True)
print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True, random_state=123)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
xp = tf.compat.v1.placeholder(tf.float32,shape=[None,8])
yp = tf.compat.v1.placeholder(tf.float32,shape=[None,1])


w1 = tf.compat.v1.Variable(tf.random_normal([8,10], name='weight1')) # (n,10) = (n,2)에 * (2,10)이 곱해져야 나옴
b1 = tf.compat.v1.Variable(tf.zeros([10], name='bias'))
layer1 = tf.compat.v1.matmul(xp, w1) + b1        #(N,10)

# layer2 : model.add(Dense(9))
w2 = tf.compat.v1.Variable(tf.random_normal([10,9], name='weight2')) # (n,10) 에 (10,9)를 곱해 (n,9)
b2 = tf.compat.v1.Variable(tf.zeros([7], name='bias'))
layer2 = tf.compat.v1.matmul(layer1,w2) + b2    #(N, 9)

# layer3 : model.add(Dense(8))
w3 = tf.compat.v1.Variable(tf.random_normal([9,8], name='weight3')) # (n,9) 에 (9,8)를 곱해 (n,8)
b3 = tf.compat.v1.Variable(tf.zeros([8], name='bias'))
layer3 = tf.compat.v1.matmul(layer2,w3) + b3    #(N, 8)

# layer4 : model.add(Dense(7))
w4 = tf.compat.v1.Variable(tf.random_normal([8,7], name='weight4')) # (n,9) 에 (9,8)를 곱해 (n,8)
b4 = tf.compat.v1.Variable(tf.zeros([7], name='bias'))
layer4 = tf.compat.v1.matmul(layer3,w4) + b4    #(N, 7)

# output_layer : model.add(Dense(1), activation='sigmoid)
w5 = tf.compat.v1.Variable(tf.random_normal([7,1], name='weight5')) # (n,9) 에 (9,8)를 곱해 (n,8)
b5 = tf.compat.v1.Variable(tf.zeros([1], name='bias'))
hypothesis = tf.compat.v1.matmul(layer4, w5) + b5   #(N, 1)


loss = tf.reduce_mean(tf.compat.v1.square(hypothesis-yp))
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0008)
train = optimizer.minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs = 15001
for step in range(epochs):
    cost_val,_ = sess.run([loss,train],
                          feed_dict={xp:x_train,yp:y_train})
    if step %100 == 0 :
        print(step, "loss:",cost_val)
        
        from sklearn.metrics import r2_score
predict = sess.run(hypothesis, feed_dict={xp: x_test})
r2 = r2_score(y_test, predict)
print("R2:", r2)
        
sess.close()