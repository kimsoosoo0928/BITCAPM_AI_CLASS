# 실습 
# 덧셈 
# 뺼셈
# 곱셈
# 나눗셈

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)
node3 = tf.add(node1, node2)
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.divide(node1, node2)

print(node3)
print(node4)
print(node5)
print(node6)
# Tensor("Add:0", shape=(), dtype=float32)
# Tensor("Sub:0", shape=(), dtype=float32)
# Tensor("Mul:0", shape=(), dtype=float32)
# Tensor("truediv:0", shape=(), dtype=float32)

sess = tf.Session()
print('node1, node2 : ', sess.run([node1, node2]))
print('sess.run(node3 : ', sess.run(node3))
print('sess.run(node4 : ', sess.run(node4))
print('sess.run(node5 : ', sess.run(node5))
print('sess.run(node6 : ', sess.run(node6))

