# y = wx + b

import tensorflow as tf
from tensorflow.python.client.session import Session
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

W = tf.Variable(1, dtype=tf.float32) # random initial number
b = tf.Variable(1, dtype=tf.float32)

hyperthesis = x_train * W + b
# hyperthesis : 통상적인 y값의 명칭 =. f(x) = wx + b

loss = tf.reduce_mean(tf.square(hyperthesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print('step :',step, 'loss :', sess.run(loss), 
                'W :', sess.run(W), 'b :', sess.run(b))