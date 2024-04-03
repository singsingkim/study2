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

'''

import tensorflow as tf
from sklearn.datasets import load_digits
tf.compat.v1.set_random_seed(777)

# 1 데이터
data, target = load_digits(return_X_y=True)
x_data = data
y_data = target
print(x_data.shape, y_data.shape)   # (1797, 64) (1797,)
print(y_data)   # [0 1 2 ... 8 9 8]


# 2 모델
# model.add(Dense(10, input_dim=2))
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64, 64]), name='weight1')
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
# layer4 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 7)
layer4 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 7)

# 아웃풋 레이어 : model.add(Dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 10]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias5')
hypothesis = tf.compat.v1.nn.softmax(tf.compat.v1.matmul(layer4, w5) + b5)   # (N, 1)


# [실습] 맹그러바
# m02_5 번과 똑같은 레이어로 구성

# 2 모델
# hypothesis = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(x, w) + b)

# 3-1 컴파일
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ cross_crossentropy ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
loss = tf.compat.v1.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))


# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predicted = tf.argmax(hypothesis , 1)
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
    print('ACC : \n', acc)
        
'''


import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder, StandardScaler
tf.compat.v1.set_random_seed(777)

# 1 데이터
data, target = load_digits(return_X_y=True)
x_data = data

# 스케일
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 라벨을 원-핫 인코딩
enc = OneHotEncoder()
y_data = enc.fit_transform(target.reshape(-1, 1)).toarray()
print(x_data.shape, y_data.shape)   # (1797, 64) (1797, 10)

# 2 모델
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 64])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64, 64]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1    # (N, 64)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64, 32]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([32]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2   # (N, 32)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([32, 16]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name='bias3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3   # (N, 16)

w4 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 8]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name='bias4')
layer4 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 8)

w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 10]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias5')
logits = tf.compat.v1.matmul(layer4, w5) + b5   # (N, 10)
hypothesis = tf.compat.v1.nn.softmax(logits)   # (N, 10)

# 3-1 컴파일
# softmax_cross_entropy_with_logits 함수 사용
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

predicted = tf.argmax(hypothesis, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.argmax(y, 1)), dtype=tf.float32))

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for step in range(5001):
        cost_val, _ = sess.run([loss, train], feed_dict={x: x_data, y: y_data})
        
        if step % 200 == 0:
            print(step, cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})
    
    print('훈련값 : \n', hypo)
    print('예측값 : \n', pred)
    print('ACC : ', acc)
