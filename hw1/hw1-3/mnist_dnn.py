from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm 
import tensorflow as tf
import pickle
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#sess = tf.InteractiveSession()
## set up variables
input_units = 784
h1_units = 512
h2_units = 128
W1 = tf.Variable(tf.truncated_normal([input_units, h1_units], stddev=0.1)) # 初始化參數
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
b2 = tf.Variable(tf.zeros([h2_units]))
W3 = tf.Variable(tf.zeros([h2_units, 10]))  # 前面是input_dim 後面是想output之dimension
b3 = tf.Variable(tf.zeros([10]))

## set up place-holder
x = tf.placeholder(tf.float32, [None, input_units]) # input_placeholder
keep_prob = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, [None, 10]) # label_placeholder

## set hidden-layer
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob) # dropout
hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
y = tf.nn.softmax(tf.matmul(hidden2, W3) + b3) # output

## cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # what reduction_indices ?
#train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in tqdm(range(1000)):
        batch_xs, batch_ys = mnist.train.next_batch(200)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})

    ## correct, evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1})

    print('The accuracy on testing set:', acc)
    #W1_weights = sess.run(W1)
    weightss = sess.run(tf.trainable_variables())
    print(weightss)
    with open('weights_adam.pickle', 'wb') as mysavedata:
        pickle.dump(weightss, mysavedata)
