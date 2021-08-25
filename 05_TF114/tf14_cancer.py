# practice

# r2 scroe

from sklearn.datasets import load_breast_cancer
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (442, 10) (442,)

x = tf.placeholder(tf.float32, shape=[None,10])
y = tf.placeholder(tf.float32, shape=[None,1])

# make it !

