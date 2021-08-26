from re import X
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import Xdivy
tf.set_random_seed(66)

x_data = [[1, 2, 1, 1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,6,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x = tf.placeholder(tf.float32, shape=[None,4])
y = tf.placeholder(tf.float32, shape=[None,3])

w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.random_normal([1,3]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)


# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
         _, cost_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
         if step % 200 == 0:
             print(step, cost_val)

    #predict
    results = sess.run(hypothesis, feed_dict={x:[[1, 11, 7, 9]]})
    print(results, sess.run(tf.argmax(results, 1)))

