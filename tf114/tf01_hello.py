import tensorflow as tf
print(tf.__version__)   # 1.14.0

print('hello world')

# 텐서1 에서 중요한 세 가지
# 컨스턴트(상수), 배리어블(변수), 프레이스홀더
hello = tf.constant('hello world')
# Tensor("Const:0", shape=(), dtype=string)
# 텐서가 출력 됌
print(hello)

sess = tf.Session()
print(sess.run(hello))
# b'hello world'