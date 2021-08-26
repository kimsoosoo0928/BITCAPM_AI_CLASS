import tensorflow as tf 
tf.set_random_seed(66)
#1.데이터
x_data = [[0,0], [0,1], [1,0], [1,1]] # 4x2
y_data = [[0],[1],[1],[0]] # 4X1

#2.모델구성
x = tf.placeholder(tf.float32, shape=[None, 2]) 
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) 
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001) # 러닝레이트가 너무 크면 nan이 나온다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
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