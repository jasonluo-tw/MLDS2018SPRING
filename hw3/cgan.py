import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class image_gan():
    def __init__(self, batch_size, z_dimension=100):
        self.z_dim = z_dimension
        self.batch_size = batch_size
        self.x_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x_placeholder')
        # x_placeholder is for feeding imput images to the discriminator
    
        self.z_placeholder = tf.placeholder(tf.float32, [None, self.z_dim], name='z_placeholder')
        # z_placeholder is for feeding input noise to the generator
        
        self.text_true = tf.placeholder(tf.float32, [None, 120])
        self.text_wrong = tf.placeholder(tf.float32, [None, 120])

        self.output_height = 64
        self.output_width = 64

    def build_model(self):
        self.Gz = self.generator(self.z_placeholder, self.text_true)
        # Gz holds the generated images

        self.Dx = self.discriminator(self.x_placeholder, self.text_true)
        # real images, real texts

        self.Dw = self.discriminator(self.x_placeholder, self.text_wrong, reuse_variables=True)
        # real images, fake texts

        self.Dg = self.discriminator(self.Gz, self.text_true, reuse_variables=True)
        # fake images, real texts
        
        # Two Loss Functions for discriminator
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dx, labels = tf.ones_like(self.Dx)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dg, labels = tf.zeros_like(self.Dg)))
        self.d_loss_wrong_text = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dw, labels = tf.zeros_like(self.Dw)))
        
        self.d_total_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_wrong_text
        # Loss function for generator
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.Dg, labels = tf.ones_like(self.Dg)))


        # Get the varabiles for different network
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'd_' in var.name]
        g_vars = [var for var in tvars if 'g_' in var.name]

        # Train approch for the discriminator
        self.d_trainer = tf.train.AdamOptimizer(0.0003, beta1=0.5).minimize(self.d_total_loss, var_list=d_vars)

        # Train approch for the generator
        self.g_trainer = tf.train.AdamOptimizer(0.0003, beta1=0.5).minimize(self.g_loss, var_list=g_vars)

    def generate_image(self, sess, z, texts):
        input_data = {}
        input_data[self.z_placeholder] = z
        input_data[self.text_true] = texts
        images = sess.run(self.Gz, feed_dict=input_data)
        images = images.reshape([-1, 64, 64])

        return images

    def train_discriminator(self, sess, real_images, z, right_texts, wrong_texts):
        input_data = {}
        input_data[self.z_placeholder] = z
        input_data[self.x_placeholder] = real_images
        input_data[self.text_true] = right_texts
        input_data[self.text_wrong] = wrong_texts
        _, d_total_loss = sess.run([self.d_trainer, self.d_total_loss], feed_dict=input_data)

        return d_total_loss

    def train_generator(self, sess, z, texts):
        input_data = {}
        input_data[self.z_placeholder] = z
        input_data[self.text_true] = texts
        _, gLoss = sess.run([self.g_trainer, self.g_loss], feed_dict=input_data)
        
        return gLoss
    
    def weights(self, name, shape, std_weight=0.08, b_std=0.0):
        name = name.lower()
        b_name = name.replace('w', 'b')
    
        W_ = tf.get_variable(name, shape, 
                initializer=tf.truncated_normal_initializer(stddev=std_weight))
        if name[2:4] == 'de':
            B_ = tf.get_variable(b_name, shape[-2], initializer=tf.constant_initializer(b_std))
        else:
            B_ = tf.get_variable(b_name, shape[-1], initializer=tf.constant_initializer(b_std))

        return W_, B_

    def conv2d(self, x, W, stride=[1, 1, 1, 1]):
        return tf.nn.conv2d(input=x, filter=W, strides=stride, padding='SAME')
    
    def deconv2d(self, x, W, output_shape):
        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, 2, 2, 1])
        return deconv

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    def discriminator(self, images, texts, reuse_variables=None):
        
        d_dim = 64 
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
            # deal with text_inputs
            d_w0, d_b0 = self.weights('d_w0', [120, 256], 0.08, 0.0)
            text_emb = tf.nn.relu(tf.matmul(texts, d_w0) + d_b0)
            text_emb = tf.reshape(text_emb, [-1, 1, 1, 256])
            
            # First convolutional and pool layers
            d_w1, d_b1 = self.weights('d_w1', [5, 5, 3,  d_dim])
            d1 = self.conv2d(images, d_w1, [1, 2, 2, 1])
            d1 = tf.contrib.layers.batch_norm(d1, epsilon=1e-5, scope='bdn1')
            d1 = tf.nn.leaky_relu(d1 + d_b1)
    
            # Sceond conv and pool
            # This finds64 different 5 x 5 pixels
            d_w2, d_b2 = self.weights('d_w2', [5, 5, d_dim, d_dim])
            d2 = self.conv2d(d1, d_w2, [1, 2, 2, 1])
            d2 = tf.contrib.layers.batch_norm(d2, epsilon=1e-5, scope='bdn2')
            d2 = tf.nn.leaky_relu(d2 + d_b2)
            
            # Third conv
            d_w3, d_b3 = self.weights('d_w3', [5, 5, d_dim, d_dim*2])
            d3 = self.conv2d(d2, d_w3, [1, 2, 2, 1])
            d3 = tf.contrib.layers.batch_norm(d3, epsilon=1e-5, scope='bdn3')
            d3 = tf.nn.leaky_relu(d3 + d_b3)

            # Forth conv 3x3
            d_w4, d_b4 = self.weights('d_w4', [3, 3, d_dim*2, d_dim*4])
            d4 = self.conv2d(d3, d_w4, [1, 1, 1, 1])
            d4 = tf.contrib.layers.batch_norm(d4, epsilon=1e-5, scope='bdn4')
            d4 = tf.nn.leaky_relu(d4 + d_b4)
            
            # deal with texts
            text_emb = tf.tile(text_emb, [1, 8, 8, 1])
            d4 = tf.concat([d4, text_emb], 3)

            # Fifth conv 3x3
            d_w5, d_b5 = self.weights('d_w5', [3, 3, 512, d_dim*8])
            d5 = self.conv2d(d4, d_w5, [1, 2, 2, 1])
            d5 = tf.contrib.layers.batch_norm(d5, epsilon=1e-5, scope='bdn5')
            d5 = tf.nn.leaky_relu(d5 + d_b5)
            
            # Linear
            index1 = d5.get_shape()[1]
            d6 = tf.reshape(d5, [-1, index1 * index1 * d_dim * 8])
            
            d_w6, d_b6 = self.weights('d_w6', [index1 * index1 * d_dim * 8, 1])
            d6 = tf.matmul(d6, d_w6) + d_b6

            # self.d4 does not pass through activation function
            return d6


    def generator(self, z, texts):

        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        s_h32, s_w32 = conv_out_size_same(s_h16, 2), conv_out_size_same(s_w16, 2)
        
        #  dimension
        img_size = s_h16 * s_w16
        g_dim = 64
        
        ## deal with text_input
        g_w0, g_b0 = self.weights('g_w0', [120, 256], 0.08, 0.0)
        text_emb = tf.nn.relu(tf.matmul(texts, g_w0) + g_b0)
        
        z_text = tf.concat((z, text_emb), 1)

        g_w1, g_b1 = self.weights('g_w1', [(self.z_dim + 256), img_size * g_dim * 8], 0.08, 0.02)
        g1 = tf.matmul(z_text, g_w1) + g_b1
        g1 = tf.reshape(g1, [-1, s_h16, s_w16, g_dim * 8])
        g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
        g1 = tf.nn.leaky_relu(g1)

        # Generate g_dim * 8 features
        g_w2, g_b2 = self.weights('g_dew2', [3, 3, g_dim * 4, g_dim * 8], 0.08, 0.02)
        g2 = self.deconv2d(g1, g_w2, [self.batch_size, s_h8, s_w8, g_dim*4])
        g2 = g2 + g_b2
        g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
        g2 = tf.nn.leaky_relu(g2)
    
        # Generate g_dim * 4 features
        g_w3, g_b3 = self.weights('g_dew3', [3, 3, g_dim * 2, g_dim * 4], 0.08, 0.02)
        g3 = self.deconv2d(g2, g_w3, [self.batch_size, s_h4, s_w4, g_dim*2])
        g3 = g3 + g_b3
        g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
        g3 = tf.nn.leaky_relu(g3)
   
        # Generate g_dim * 2 feature
        g_w4, g_b4 = self.weights('g_dew4', [3, 3, g_dim * 1, g_dim * 2], 0.08, 0.02)
        g4 = self.deconv2d(g3, g_w4, [self.batch_size, s_h2, s_w2, g_dim * 1])
        g4 = g4 + g_b4
        g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope='bn4')
        g4 = tf.nn.leaky_relu(g4)

        # Generate g_dim * 2
        g_w5, g_b5 = self.weights('g_dew5',[3, 3, 3, g_dim * 1], 0.08, 0.02)
        g5 = self.deconv2d(g4, g_w5, [self.batch_size, s_h, s_w, 3])
        g5 = g5 + g_b5
        #g5 = tf.contrib.layers.batch_norm(g5, epsilon=1e-5, scope='bn5')
        g5 = tf.nn.tanh(g5)

        # Last
        #g_w6, g_b6 = self.weights('g_dew6',[3, 3, 3, g_dim * 1], 0.02, 0.02)
        #g6 = self.deconv2d(g5, g_w6, [self.batch_size, s_h, s_w, 3])
        #g6 = g6 + g_b6
        #g6 = tf.nn.tanh(g6)
        
        # Dimension of g6: batch_size x 64 x 64 x 3 
        return g5


if __name__ == '__main__':

    model = image_gan(64)
    model.build_model()
    print('True')
