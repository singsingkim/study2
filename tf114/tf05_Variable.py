import tensorflow as tf
print(tf.__version__)

sess = tf.Session()

a = tf.Variable([2], dtype=tf.float32)
b = tf.Variable([3], dtype=tf.float32)

# print(sess.run(a + b))
# 에러
# 텐서에서는 변수정의했을때 꼭 초기화를 해주어야 한다.
# 초기화 정의시키고 run 시켜주어야 한다
init = tf.compat.v1.global_variables_initializer()  # 초기화
sess.run(init)  # run

print(sess.run(a + b))

sess.close()