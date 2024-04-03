from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(777)
from sklearn.preprocessing import MinMaxScaler

# 1 데이터
data, target = load_breast_cancer(return_X_y=True)

x_data , y_data = data, target

print(x_data.shape, y_data.shape)   # (569, 30) (569,)


keep_prob = tf.placeholder(tf.float32)


scaler = MinMaxScaler()
x_data = scaler.fit_transform(x_data)

x = tf.compat.v1.placeholder(tf.float32, shape=[None, 30])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, ])

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([30, 10]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1    # (N, 10)
print(layer1.shape) # (?, 10)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 9]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([9]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2   # (N, 9)
layer2 = tf.compat.v1.nn.dropout(layer2, keep_prob=keep_prob)
# 훈련할때에 드롭아웃 적용이되고 0.5가 적용이 된게 평가할때도 0.5가 적용이된다. 평가 할때는 전체(1.0)가 적용이 되어야 한다.


w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([9, 8]), name='weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([8]), name='bias3')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3   # (N, 8)

w4= tf.compat.v1.Variable(tf.compat.v1.random_normal([8, 7]), name='weight4')
b4 = tf.compat.v1.Variable(tf.compat.v1.zeros([7]), name='bias4')
layer4 = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer3, w4) + b4)   # (N, 7)

# 아웃풋 레이어 : model.add(Dense(1), activation='sigmoid')
w5 = tf.compat.v1.Variable(tf.compat.v1.random_normal([7, 1]), name='weight5')
b5 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias5')
hypothesis = tf.compat.v1.sigmoid(tf.compat.v1.matmul(layer4, w5) + b5)   # (N, 1)


# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ binary_crossentropy ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
loss = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1-hypothesis))

# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-5)
# train = optimizer.minimize(loss)
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        cost_val, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data, keep_prob:0.2})
        
        if step % 200 == 0:
            print(step, cost_val)
            
    hypo, pred, acc = sess.run([hypothesis, predicted, accuracy],
                               feed_dict={x:x_data, y:y_data, keep_prob:1.0})
    
    print('훈련값 : \n', hypo)
    print('예측값 : \n', pred)
    print('ACC : \n', acc)
        
