import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# 1 데이터
x_train = [1,2,3]
y_train = [1,2,3]
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)

w = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')
b = tf.compat.v1.Variable([10], dtype=tf.float32, name='weight')


# 2 모델
hypothesis = x * w + b


# 3-1 컴파일 // model.compile(loss='mse', ooptimizer='sgd')
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

########### ↓↓↓↓↓↓↓↓↓↓↓ 옵티마이저 ↓↓↓↓↓↓↓↓↓↓↓↓ ###########
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)   # GD = 경사하강법(loss 최소화) / lr = 보폭을 얼마나 잡는지
# train = optimizer.minimize(loss)    # loss 를 최소화 시키겠다.
lr = 0.1

gradient = tf.reduce_mean((x * w + b- y) * x)

descent = w - lr * gradient
update = w.assign(descent)
# 여기까지가 경사하강법
########### ↑↑↑↑↑↑↑↑↑↑ 옵티마이저 ↑↑↑↑↑↑↑↑↑↑↑ ###########

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

w_hist = []
loss_hist = []

for step in range(21):
    _, loss_v, w_v = sess.run([update, loss, w], feed_dict={x:x_train, y:y_train})
    print(step, '\t', loss_v, '\t', w_v)
    
    w_hist.append(w_v)
    loss_hist.append(loss_v)
sess.close()



print('====================== w history =========================')
print(w_hist)
print('===================== loss history =========================')
print(loss_hist)

plt.plot(loss_hist)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()