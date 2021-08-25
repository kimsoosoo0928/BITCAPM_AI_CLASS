# 표현방식의 차이

import tensorflow as tf
tf.compat.v1.set_random_seed(777)

x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = x * W + b

# 실습
# tf09 1번의 방식 3가지로 hypothesis를 출력하시오.



# 표현1
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(x * W + b) # 
print("aaa : ", aaa) # aaa :  [1.3       1.6       1.9000001]
sess.close()

# 표현2
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = (x * W + b).eval()
print("bbb : ", bbb) # bbb :  [1.3       1.6       1.9000001]
sess.close()

# 표현3
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc = (x * W + b).eval(session=sess)
print("ccc : ", ccc) # ccc :  [1.3       1.6       1.9000001]
sess.close()