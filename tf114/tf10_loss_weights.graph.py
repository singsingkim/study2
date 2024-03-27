import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)

# x = [1,2,3]
# y = [1,2,3]

# ## 손 계산 실습 ###
x = [1,2]
y = [1,2]

w = tf.compat.v1.placeholder(tf.float32)

hypothesis = x * w

loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse

w_hist = []
loss_hist = []

with tf.compat.v1.Session() as sess:
    for i in range(-30, 50):
        curr_w = i
        curr_loss = sess.run(loss, feed_dict={w:curr_w})
        
        w_hist.append(curr_w)
        loss_hist.append(curr_loss)
        
        
print('====================== w history =========================')
print(w_hist)
print('===================== loss history =========================')
print(loss_hist)

plt.plot(w_hist, loss_hist)
plt.xlabel('weights')
plt.ylabel('loss')
plt.show()