from tensorflow.examples.tutorials.mnist import input_data
import csv
from math import floor
from tqdm import tqdm 
import tensorflow as tf
import pickle
import numpy as np
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--save_weights", type=bool, default=False, help="save weights or not")
ap.add_argument("-al", "--save_al", type=bool, default=False, help="save acc loss or not")
ap.add_argument("-bs", "--batch_size", type=int, default=1024, help="batch size")
args = vars(ap.parse_args())

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
y_ = tf.placeholder(tf.float32, [None, 10]) # label_placeholder

## set hidden-layer
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
y = tf.nn.softmax(tf.matmul(hidden2, W3) + b3) # output

## cross_entropy
#cross_entropy = -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]) # what reduction_indices ?
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,10.0)), reduction_indices=[1]))

mean_cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,10.0)), reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
#train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

init = tf.global_variables_initializer()

batch_size = args["batch_size"]
epochs = 100 * int(floor(len(mnist.train.images) / batch_size))
#step_num = int(floor(len(mnist.train.images) / batch_size))

with tf.Session() as sess:
    sess.run(init)
    for i in tqdm(range(epochs)):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Train with batch
        #for j in range(step_num):
        #    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    ## count numbers of all the parameters in the model
    #para_nums = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    ## correct, evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    ## accuracy
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    acc_train = sess.run(accuracy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})
    ## loss
    loss = sess.run(mean_cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    loss_train = sess.run(mean_cross_entropy, feed_dict={x: mnist.train.images, y_: mnist.train.labels})

    #grads_wrt_input = sess.run(tf.gradients(cross_entropy, x), feed_dict={x: mnist.train.images, y_: mnist.train.labels})
    grads_wrt_input = sess.run(tf.gradients(cross_entropy, x), feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    #sensitivity = np.sqrt(np.sum(np.mean(grads_wrt_input[0], axis=0) ** 2))
    sensitivity = np.sqrt(np.sum(grads_wrt_input[0] ** 2))
    print(sensitivity)
    print('The accuracy on testing set:', acc)
    print('The accuracy on training set:', acc_train)

    weightss = sess.run(tf.trainable_variables())

    if args["save_weights"] is True:
        with open('weights_adam.pickle', 'wb') as mysavedata:
            pickle.dump(weightss, mysavedata)
    if args["save_al"] is True:
        file_out = open('./loss_acc_seni_test.csv', 'a+')
        s = csv.writer(file_out, delimiter=',', lineterminator='\n')
        s.writerow([batch_size, loss_train, loss, acc_train, acc, sensitivity])
        file_out.close()
