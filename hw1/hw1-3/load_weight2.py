from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np

input_units = 784

with open('./weights.pickle', 'rb') as myfile:
    weights0 = pickle.load(myfile)

with open('./weights_adam.pickle', 'rb') as myfile2:
    weights1 = pickle.load(myfile2)


W1 = tf.placeholder(tf.float32, [784, 512])
B1 = tf.placeholder(tf.float32, [512])

W2 = tf.placeholder(tf.float32, [512, 128])
B2 = tf.placeholder(tf.float32, [128])

W3 = tf.placeholder(tf.float32, [128, 10])
B3 = tf.placeholder(tf.float32, [10])

## set up place-holder
x = tf.placeholder(tf.float32, [None, input_units])
y_ = tf.placeholder(tf.float32, [None, 10])
## set DNN
hidden1 = tf.nn.relu(tf.matmul(x, W1) + B1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + B2)
y = tf.nn.softmax(tf.matmul(hidden2, W3) + B3) # output
## cross_entropy
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # what reduction_indices ?
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)), reduction_indices=[1]))
## accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## data process
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

init = tf.global_variables_initializer()

acc_list = []
loss_list = []
xx = []

with tf.Session() as sess:
    sess.run(init)

    alpha = -1.0
    while alpha <= 2.0:
        #print(alpha)
        theta = []
        for idx, wei0 in enumerate(weights0):
            theta.append([])
            theta[idx] = (1 - alpha) * wei0 + alpha * weights1[idx]
            #print(theta)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, 
            W1: theta[0], B1: theta[1], W2: theta[2], B2: theta[3], W3: theta[4], B3: theta[5]})
        loss = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, 
            W1: theta[0], B1: theta[1], W2: theta[2], B2: theta[3], W3: theta[4], B3: theta[5]})
        #print('The accuracy, loss on testing set:', acc, loss)
        xx.append(alpha)
        acc_list.append(acc)
        loss_list.append(loss)
        alpha += 0.1

fig = plt.figure()
ax = fig.add_subplot(111)

ax2 = ax.twinx()

p1, = ax.plot(xx, acc_list, 'g-', label='accuracy')
p2, = ax2.plot(xx, loss_list, 'b-', label='loss')
fig.legend(loc=2, bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

ax.set_xlabel('alpha')
ax.set_ylabel('accuracy', color='g')
ax2.set_ylabel('cross entropy', color='b')



plt.show()
