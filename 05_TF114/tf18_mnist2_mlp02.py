# import 
from keras.datasets import mnist
from scipy.stats.morestats import Std_dev
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

# 1. data 
datasets = tf.keras.datasets.mnist # data def.
 
(x_train, y_train), (x_test, y_test) = datasets.load_data() # mnist data pull 

x_train = x_train.reshape(60000,28*28*1)/255. # DNN을 위해 차원 축소, (?)왜 255로 나눠주는지 잘 모르겠음 
x_test = x_test.reshape(10000,28*28*1)/255. 


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray()
y_test = one.transform(y_test).toarray()
# y에 대한 one hot encoding -> 왜해줬던거지? 기억이 나질 않는다..

# 2. model * tensorflow v1 model은 v2에 비해 난해함...
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])

# 2_1 hidden layer 
W0 = tf.Variable(tf.random.normal([28*28,270], stddev=0.1, name='weight'))
b0 = tf.Variable(tf.random.normal([1,270], stddev=0.1, name='bias'))
layer0 = tf.nn.relu(tf.matmul(x,W0) + b0)
layer0 = tf.nn.dropout(layer0, keep_prob=0.1)

W1 = tf.Variable(tf.random.normal([270,100], stddev=0.1, name='weight'))
b1 = tf.Variable(tf.random.normal([1,100], stddev=0.1, name='bias'))
layer1 = tf.nn.relu(tf.matmul(x,W1) + b1)
# layer0 = tf.nn.dropout(layer0, keep_prob=0.1)

# 2_2 ouput layer
W2 = tf.Variable(tf.random.normal([100,10], stddev=0.1, name='weight'))
b2 = tf.Variable(tf.random.normal([1,10], stddev=0.1, name='bias'))
layer2 = tf.nn.relu(tf.matmul(x,W1) + b2)

# categorical_crossentropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(layer2), axis=1)) 

optimizer = tf.train.AdamOptimizer(learning_rate=0.0008).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epochs in range(131):
        cost_val, _ = sess.run([cost, optimizer],
            feed_dict={x:x_train, y:y_train})
        if epochs % 3 == 0:
            print(epochs, "cost :", cost_val)

    predict = sess.run(layer2, feed_dict={x:x_test}) # (?)sess.run, (?)feed_dict 
    print(sess.run(tf.argmax(predict, 1))) # (?) argmax ?

    y_pred = sess.run(layer2, feed_dict={x:x_test})
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_pred, axis=1)
    print('acc_score : ', accuracy_score(y_test, y_pred))


