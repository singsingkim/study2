# 실습
# lr 수정해서 epoch 101번 이하로 줄여서
# step = 100 이하, w = 1.99 이하, b = 0.99 이하

import tensorflow as tf
tf.set_random_seed(777)
print(tf.__version__)

import matplotlib.pyplot as plt

# 1. 데이터
x_data = [1,2,3,4,5]
y_data = [3,5,7,9,11]        # 웨이트2, 바이어스1 (x*2+1)
x = tf.compat.v1.placeholder(tf.float32, shape=[None])
y = tf.compat.v1.placeholder(tf.float32, shape=[None])



# y = wx + b
# w = tf.Variable(111, dtype=tf.float32)  # weight
# b = tf.Variable(0, dtype=tf.float32)    # bias
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규분포
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)  # 정규분포
print(w)    # <tf.Variable 'Variable:0' shape=(1,) dtype=float32_ref>

# sess = tf.compat.v1.Session()
# sess.run(tf.global_variables_initializer()) # 변수 초기화
# print(sess.run(w))  # [-1.5080816] - 시드고정했을때 고정됌


# 2. 모델 
# y = wx + b
# hypothesis = w * x + b   # 이거 아니다. 이제는 말할 수 있다.
hypothesis = x * w + b      # 실질적인 predict  옵티마이저 사용하기 위해

# 3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))    # mse(차이의 제곱의 평균)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0823)   # GD = 경사하강법(loss 최소화) / lr = 보폭을 얼마나 잡는지
train = optimizer.minimize(loss)    # loss 를 최소화 시키겠다.
# model.compile(loss='mse', optimizer='sgd') 여기까지 진행한것

# 3-2. 훈련
# sess = tf.compat.v1.Session()
# 2번 파일과 같이 close 를 잡거나, with 로 범위 지정하여 자동 close

loss_val_list = []
w_val_list = []

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer()) # 변수를 초기화 시키겠다.

    # model.fit
    epochs = 101
    for step in range(epochs):
        # sess.run(train)         # 여기까지는 단순히 1 에포
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                                             feed_dict={x:x_data, y:y_data})
        
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val)
            # print(step, sess.run(loss), sess.run(w), sess.run(b))   # verbose 와 model.weight 에서 확인했던 놈들.
    # sess.close()        



# 4 예측
##############################[실습]########################################
x_pred = [6,7,8]
# 예측값을 뽑아봐
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])
# 결과치
# print('[6,7,8]의 예측 : ')
############################################################################

'''
내가 실습해본거. 망했음.
y_pred = x_pred * x_test + b

print(sess.run(hypothesis, feed_dict={x:x_pred, w:x_test}))
'''


# y_predict = xw + b

with tf.compat.v1.Session() as sess:

    # 1 파이썬방식 해결
    y_pred = x_pred * w_val + b_val
    print('[6,7,8]의 예측 : ', y_pred)

    # 2 placeholder 해결
    y_pred2 = x_test * w_val + b_val
    print('[6,7,8]의 예측 : ', sess.run(y_pred2, feed_dict={x_test:x_pred}), w_val, b_val)

print(loss_val_list)
print(w_val_list)
   
   
   
# subplot을 사용하여 여러 그래프를 한 번에 표시
# 그래프를 4분할하는 서브플롯 생성
fig, axs = plt.subplots(2, 2)

# 첫 번째 그래프: 손실 함수 값 그래프
axs[0, 0].plot(loss_val_list)
axs[0, 0].set_xlabel('epochs')
axs[0, 0].set_ylabel('loss')

# 두 번째 그래프: 가중치 값 그래프
axs[0, 1].plot(w_val_list)
axs[0, 1].set_xlabel('epochs')
axs[0, 1].set_ylabel('weights')

# 세 번째 그래프: 가중치와 손실 함수 값의 산점도
axs[1, 0].scatter(w_val_list, loss_val_list)
axs[1, 0].set_xlabel('weights')
axs[1, 0].set_ylabel('loss')

# 네 번째 그래프: 공백
axs[1, 1].axis('off')

# 서브플롯 레이아웃 조정
plt.tight_layout()

# 그래프 출력
plt.show()
