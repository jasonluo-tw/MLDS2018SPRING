import numpy as np
import cv2
import os
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from cgan import image_gan
#from cwgan import image_gan
from read_data_cgan import read_imgs
from read_data_cgan import setup_dicts
import math

print("Now train cgan ...")
####### Define some variables

batch_size = 64
z_dimension = 100
# for test images
z_batch0 = np.random.normal(size=[batch_size, z_dimension]) 
tags00 = [1] + [0] * 119
tags0 = [tags00 for i in range(batch_size)]
loss_lists = []
####################
## init the model ##
gan_model = image_gan(batch_size)
gan_model.build_model()
saver = tf.train.Saver()

## read dataset
datasets = read_imgs()
#datasets.read_source2('./dataset/AnimeDataset/faces/', True)
datasets.read_source1('./dataset/extra_data/images/')
dicts_ = setup_dicts('./dataset/extra_data/tags.csv')
datasets.read_source1_tags('./dataset/extra_data/tags.csv', dicts_)
datasets.shuffle_data()

print(datasets.length)

## Start graph
with tf.Session() as sess:
    
    # init global variables
    sess.run(tf.global_variables_initializer())

    # Pre-train discriminator
    print('Start pre-train discriminator')

    for i in range(51):
        z_batch = np.random.normal(size=[batch_size, z_dimension])
        real_image_batch, real_tags, fake_tags = datasets.next_batch(batch_size, True)
        d_loss_all = gan_model.train_discriminator(sess, real_image_batch, z_batch,
                real_tags, fake_tags)

        if(i % 50 == 0):
            print("dLossAll: ", d_loss_all)

    # Train generator and discriminator together
    for itera in tqdm(range(10001)):
        real_image_batch, real_tags, fake_tags = datasets.next_batch(batch_size, True)
        z_batch = np.random.normal(size=[batch_size, z_dimension])

        # Train discriminator
        for ii in range(4):
            d_loss_all = gan_model.train_discriminator(
                    sess, real_image_batch, z_batch,
                    real_tags, fake_tags)

        
        z_batch = np.random.normal(size=[batch_size, z_dimension])
        # Train generator
        for ii in range(3):
            gg_loss = gan_model.train_generator(sess, z_batch, real_tags)

        if itera % 1000 == 0:
            loss_lists.append(d_loss_all)
            print("dLossAll: ", d_loss_all)
    
            # For generate images
            testImage = gan_model.generate_image(sess, z_batch0, tags0)
            testImage = testImage.reshape([batch_size, 64, 64, 3])
            testImage = (testImage + 1) / 2.
            for jj, im in enumerate(testImage):
                plt.subplot(8, 8, jj+1)
                plt.axis('off')
                plt.imshow(im)

            #plt.show()
            plt.savefig('imgc/test_img'+str(itera)+'.png')

    saver.save(sess, './models/cgan.ckpt')
            

with open('training_lossc.txt', 'w') as f:
    for ii in loss_lists:
        f.write('d_train_loss: '+str(ii)+'\n')

