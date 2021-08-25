# practice

# r2 scroe

from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_boston()
x_data = datasets.data
y_data = datasets.target

from sklearn.model_selection import train_test_split

y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=77)

x = tf.placeholder(tf.float32, shape=[None,13])
y = tf.placeholder(tf.float32, shape=[None,1])

# make it !

w = tf.Variable(tf.random_normal([13,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x, w) + b

cost = tf.reduce_mean(tf.square(hypothesis-y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.AdamOptimizer(learning_rate=8e-1)
train = optimizer.minimize(cost)



sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
        feed_dict={x:x_train, y:y_train})
    if epochs % 200 == 0:
        print(epochs, "cost :", cost_val)


pred = sess.run(hypothesis, feed_dict={x:x_test})

r2 = r2_score(y_test, pred)

print('r2_score : ', r2)

sess.close()

# r2_score :  0.7300884014515201