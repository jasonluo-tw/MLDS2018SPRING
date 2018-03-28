from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pickle
import numpy as np

alpha = 0.5
input_units = 784
theta = []

with open('./weights.pickle', 'rb') as myfile:
    weights0 = pickle.load(myfile)

with open('./weights_adam.pickle', 'rb') as myfile2:
    weights1 = pickle.load(myfile2)

for idx, wei0 in enumerate(weights0):
    theta.append([])
    theta[idx] = (1 - alpha) * wei0 + alpha * weights1[idx]
    #print(theta)

W1 = tf.convert_to_tensor(theta[0], np.float32)
B1 = tf.convert_to_tensor(theta[1], np.float32)

W2 = tf.convert_to_tensor(theta[2], np.float32)
B2 = tf.convert_to_tensor(theta[3], np.float32)

W3 = tf.convert_to_tensor(theta[4], np.float32)
B3 = tf.convert_to_tensor(theta[5], np.float32)

## set up place-holder
x = tf.placeholder(tf.float32, [None, input_units])
y_ = tf.placeholder(tf.float32, [None, 10])
## set DNN
hidden1 = tf.nn.relu(tf.matmul(x, W1) + B1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + B2)
y = tf.nn.softmax(tf.matmul(hidden2, W3) + B3) # output
## cross_entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # what reduction_indices ?
## accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

## data process
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('The accuracy on testing set:', acc)
