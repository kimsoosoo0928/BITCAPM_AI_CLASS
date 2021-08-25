# practice

# accuracy_score

from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

x = tf.placeholder(tf.float32, shape=[None,30])
y = tf.placeholder(tf.float32, shape=[None,])

# make it !


w = tf.Variable(tf.random_normal([30,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) 
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis)) # binary_crossentropy

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001) # 러닝레이트가 너무 크면 nan이 나온다.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000000001)
train = optimizer.minimize(cost)


sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

#3.훈련
for epochs in range(101):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                feed_dict={x:x_data, y:y_data})
    if epochs % 20 == 0:
        print(epochs, "cost : ", cost_val, "\n", hy_val)

#4. 평가,예측

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted, accuracy], feed_dict={x:x_data, y:y_data})

print("예측값 : \n", hy_val,
         "\n 예측결과값 : \n", c, "\n Accuracy : \n", a)

sess.close()