# practice

# r2 scroe

from sklearn.datasets import load_boston
import tensorflow as tf
tf.set_random_seed(66)

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (506, 13) (506,)

x = tf.placeholder(tf.float32, shape=[None,13])
y = tf.placeholder(tf.float32, shape=[None,1])

# make it !

