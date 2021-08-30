import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D

tf.set_random_seed(66)

### get_variable vs Variable
### CNN, activation


# 1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255.

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# 모델구성
# 변수 초기화
##### tensorflow 1
## layer 1
w1 = tf.get_variable('w1', shape=[3, 3, 1, 32])   # [kernel_size, input의 channel(color) 수(맨 마지막), output]
layer1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')   # x, w의 차원이 같아야 함(둘이 연산되어야 하니까)    # strides : 4차원으로 잡아줘야 함
layer1_activation = tf.nn.relu(layer1)     # activation
layer1_maxpool = tf.nn.max_pool(layer1_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')     # maxpool
                                                  # (자리채우기 위함, kenelsize:(2,2), 자리채우기 위함)
##### tensorflow 2
# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', input_shape=(28, 28, 1),
#                  activation='relu'))

print(layer1_activation)    # (?, 28, 28, 32)
print(layer1_maxpool)       # (?, 14, 14, 32)



## layer 2
w2 = tf.get_variable('w2', shape=[3, 3, 32, 64])
layer2 = tf.nn.conv2d(layer1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
layer2_activation = tf.nn.selu(layer2)
layer2_maxpool = tf.nn.max_pool(layer2_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(layer2_activation)    # (?, 14, 14, 64)
print(layer2_maxpool)       # (?, 7, 7, 64)




## layer 3
w3 = tf.get_variable('w3', shape=[3, 3, 64, 128])
layer3 = tf.nn.conv2d(layer2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
layer3_activation = tf.nn.elu(layer3)
layer3_maxpool = tf.nn.max_pool(layer3_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(layer3_activation)    # (?, 7, 7, 128)
print(layer3_maxpool)       # (?, 4, 4, 128)




## layer 4
w4 = tf.get_variable('w4', shape=[2, 2, 128, 64])
layer4 = tf.nn.conv2d(layer3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
layer4_activation = tf.nn.leaky_relu(layer4)
layer4_maxpool = tf.nn.max_pool(layer4_activation, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(layer4_activation)    # (?, 3, 3, 64)
print(layer4_maxpool)       # (?, 2, 2, 64)




## Flatten
L_flat = tf.reshape(layer4_maxpool, [-1, 2*2*64])
print("플랫튼 :", L_flat)    # (?, 256)




## layer5  DNN
w5 = tf.get_variable('w5', shape=[2*2*64, 64])
b5 = tf.Variable(tf.random_normal([64]), name='b1')
layer5 = tf.matmul(L_flat, w5) + b5
layer5_activation = tf.nn.selu(layer5)
layer5_dropout = tf.nn.dropout(layer5_activation, keep_prob=0.2)

print(layer5_dropout)        # (?, 64)




## layer6  DNN
w6 = tf.get_variable('w6', shape=[64, 32])
b6 = tf.Variable(tf.random_normal([32]), name='b2')
layer6 = tf.matmul(layer5_dropout, w6) + b6
layer6_activation = tf.nn.selu(layer6)
layer6_dropout = tf.nn.dropout(layer6_activation, keep_prob=0.2)

print(layer6_dropout)        # (?, 32)




## layer7  softmax
w7 = tf.get_variable('w7', shape=[32, 10])
b7 = tf.Variable(tf.random_normal([10]), name='b3')
layer7 = tf.matmul(layer6_dropout, w7) + b7
hypothesis = tf.nn.softmax(layer7)

print(hypothesis)      # (?, 10)