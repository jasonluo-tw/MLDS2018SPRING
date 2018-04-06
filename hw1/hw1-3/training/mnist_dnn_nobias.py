from tensorflow.examples.tutorials.mnist import input_data
import csv
from math import floor
from tqdm import tqdm 
import tensorflow as tf
import pickle
import numpy as np
import argparse
from scipy.linalg import block_diag

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--save_weights", type=bool, default=False, help="save weights or not")
ap.add_argument("-al", "--save_al", type=bool, default=False, help="save acc loss or not")
ap.add_argument("-bs", "--batch_size", type=int, default=1024, help="batch size")
ap.add_argument("-ps", "--epochs", type=int, default=100, help="batch size")
args = vars(ap.parse_args())

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#sess = tf.InteractiveSession()
## set up variables
input_units = 784
h1_units = 32
h2_units = 32

W1_ = tf.Variable(tf.truncated_normal([input_units * h1_units], stddev=0.1))
W1 = tf.reshape(W1_, [input_units, h1_units])
#W1 = tf.Variable(tf.truncated_normal([input_units, h1_units], stddev=0.1)) # 初始化參數

W2_ = tf.Variable(tf.truncated_normal([h1_units * h2_units], stddev=0.1))
W2 = tf.reshape(W2_, [h1_units, h2_units])
#W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))

W3_ = tf.Variable(tf.zeros(h2_units * 10))
W3 = tf.reshape(W3_, [h2_units, 10])
#W3 = tf.Variable(tf.zeros([h2_units, 10]))  # 前面是input_dim 後面是想output之dimension

## set up place-holder
x = tf.placeholder(tf.float32, [None, input_units]) # input_placeholder
y_ = tf.placeholder(tf.float32, [None, 10]) # label_placeholder

## set hidden-layer
hidden1 = tf.nn.relu(tf.matmul(x, W1))
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2))
y = tf.nn.softmax(tf.matmul(hidden2, W3)) # output
## cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

mean_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))

hessian = tf.hessians(cross_entropy, [W1_, W2_, W3_])
h0_ = tf.norm(hessian[0], 2)
h1_ = tf.norm(hessian[1], 2)
h2_ = tf.norm(hessian[2], 2)
max_ = tf.reduce_max([h0_, h1_, h2_])

train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()

epochs = args["epochs"]
batch_size = args["batch_size"]
step_num = int(floor(len(mnist.train.images) / batch_size))

with tf.Session() as sess:
    sess.run(init)
    for i in tqdm(range(epochs)):
        # Train with batch
        for j in range(step_num):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    ## correct, evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    ## accuracy
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
    ## loss
    loss = sess.run(mean_cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    loss_train = sess.run(mean_cross_entropy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})

    print('The accuracy on testing set:', acc)
    print('The accuracy on training set:', acc_train)
    loss_c = sess.run(cross_entropy, feed_dict={x: mnist.train.images[:500], y_: mnist.train.labels[:500]})
    L2normHes = sess.run(max_, feed_dict={x: mnist.train.images[:500], y_: mnist.train.labels[:500]})

    sharp = L2normHes * (1e-3)**2  / (2 * (1 + loss_c))
    print(sharp)
    if args["save_weights"] is True:
        with open('weights_adam.pickle', 'wb') as mysavedata:
            pickle.dump(weightss, mysavedata)
    if args["save_al"] is True:
        file_out = open('./loss_acc_sharpness.csv', 'a+')
        s = csv.writer(file_out, delimiter=',', lineterminator='\n')
        s.writerow([batch_size, loss_train, loss, acc_train, acc, sharp])
        file_out.close()
