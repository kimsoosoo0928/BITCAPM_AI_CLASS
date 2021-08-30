import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) # False
print(tf.__version__)         # 1.14.0 -> 2.4.1


# tf.set_random_seed(66)

# 1. 데이터
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32')/255

learning_rate = 0.0001
training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 2. 모델구성

# layer1
W1 = tf.compat.v1.get_variable('W1', shape=[3, 3, 1, 32])
                               # [kernel_size, input_shape_channel, output_filter]
print(W1) # (3, 3, 1, 32)

L1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,                    color
#           padding='valid', input_shape=(28, 28, 1)),          # (low, cols, channel)
#           activation='relu')
# model.add(MaxPool2D())

print(L1)         # (?, 28, 28, 32)
print(L1_maxpool) # (?, 14, 14, 32)

# layer2
W2 = tf.compat.v1.get_variable('W2', shape=[3, 3, 32, 64])
L2 = tf.nn.conv2d(L1_maxpool, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L2)         # (?, 14, 14, 64)
print(L2_maxpool) # (?,  7,  7, 64)

# layer3
W3 = tf.compat.v1.get_variable('W3', shape=[3, 3, 64, 128])
L3 = tf.nn.conv2d(L2_maxpool, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.selu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L3)         # (?, 7, 7, 128)
print(L3_maxpool) # (?, 4, 4, 128)

# layer4
W4 = tf.compat.v1.get_variable('W4', shape=[2, 2, 128, 64],)
                    # initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, W4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print(L4)         # (?, 4, 4, 64)
print(L4_maxpool) # (?, 2, 2, 64)

# Flatten
L_flat = tf.reshape(L4_maxpool, [-1, 2*2*64])
print("Flatten", L_flat)   # (?, 256)

# layer5 DNN
W5 = tf.compat.v1.get_variable('W5', shape=[2*2*64, 64],)
                    # initializer=tf.contrib.layers.xavier_initializer())
B5 = tf.Variable(tf.random.normal([64]), name='B1')
L5 = tf.matmul(L_flat, W5) + B5
L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)
print(L4)         # (?, 4, 4, 64)
print(L4_maxpool) # (?, 2, 2, 64)

# layer6 DNN
W6 = tf.compat.v1.get_variable("W6", shape=[64, 32])
B6 = tf.Variable(tf.random.normal([32]), name='B2')
L6 = tf.matmul(L5, W6) + B6
L6 = tf.nn.selu(L6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)

print(L6) # (?, 32)

# layer7 Softmax
W7 = tf.compat.v1.get_variable("W7", shape=[32, 10])
B7 = tf.Variable(tf.random.normal([10]), name='B3')
L7 = tf.matmul(L6, W7) + B7
hypothesis = tf.nn.softmax(L7)
 
print(hypothesis) # (?, 10)

# 3. 컴파일 훈련
# categorical_crossentropy
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(hypothesis), axis=1))

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# total_batch = int(len(x_train)/batch_size)

for epoch in range(training_epochs):
    avg_loss = 0

    for i in range(total_batch):      # 몇 번 도는가? 600번
        start = i * batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        avg_loss += batch_loss/total_batch
    
    print('Epoch : ', '%04d' %(epoch + 1), 'loss : {:.9f}'.format(avg_loss))

print("훈련 끝")

prediction = tf.equal(tf.compat.v1.arg_max(hypothesis, 1), tf.compat.v1.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('ACC : ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))


# ACC :  0.691