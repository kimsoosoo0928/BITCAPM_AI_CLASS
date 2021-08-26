# practice

# from tensorflow.keras.datasets import mnist
import tensorflow as tf
from keras.datasets import mnist
import numpy as np

tf.set_random_seed(66)

# 2. model
x_train = tf.placeholder(tf.float32, shape=[None, 28*28]) 
y_train = tf.placeholder(tf.float32, shape=[None, 10])

# output layer

w = tf.Variable(tf.random_normal([28*28,10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

layers1 = tf.relu(tf.matmul(x_train, w) + b)
layers2 = tf.elu(tf.matmul(x_train, w) + b)
layers3 = tf.selu(tf.matmul(x_train, w) + b)
layers4 = tf.sigmoid(tf.matmul(x_train, w) + b)
layers = tf.nn.dropout(layers4, keep_prob=0.3)


# cost = tf.reduce_mean(tf.square(hypothesis-y)) 
cost = -tf.reduce_mean(y_train*tf.log(layers)+(1-y_train)*tf.log(1-y_train)) # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001) # 러닝레이트가 너무 크면 nan이 나온다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

#3.훈련
for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                feed_dict={x:x_data, y:y_data})
    if epochs % 200 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)

#4. 평가,예측

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})

print("예측값 : \n", hy_val,
         "\n 예측결과값 : \n", c, "\n Accuracy : \n", a)

sess.close()