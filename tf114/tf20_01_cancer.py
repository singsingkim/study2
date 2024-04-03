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
from sklearn.datasets import load_breast_cancer
tf.compat.v1.set_random_seed(777)

# 1 데이터
data, target = load_breast_cancer(return_X_y=True)
x_data = data
y_data = target
print(x_data.shape, y_data.shape)   # (569, 30) (569,)
print(y_data)

# 2 모델
# model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 64]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1    # (N, 10)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64, 32]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2   # (N, 9)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32, 16]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name='bias3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3   # (N, 8)

w4= tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 8]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name='bias4')
layer4 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 7)
# layer4 = tf.compat.v1.matmul(layer3, w4) + b4   # (N, 7)

# 아웃풋 레이어 : model.add(Dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias5')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)   # (N, 1)


# [실습] 맹그러바
# m02_5 번과 똑같은 레이어로 구성

# 2 모델
# hypothesis = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1 컴파일
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ binary_crossentropy ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})
        
        if step % 200 == 0:
            print(step, cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={x:x_data, y:y_data})
    
    print('훈련값 : \n', hypo)
    print('예측값 : \n', pred)
    print('ACC : ', acc)    

     # ACC : 0.6274165
        

