import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cgan import image_gan
from read_data_cgan import setup_dicts
import math

print("Generate images")

dicts_ = setup_dicts('./dataset/extra_data/tags.csv')
## setup tags
f = open('./dataset/AnimeDataset/sample_testing_text.txt', 'r')
datas = f.readlines()
tags0 = []
for index, tagss in enumerate(datas):
    tags_0 = [0] * len(dicts_)
    tags00 = tagss.split(',')[-1].rstrip('\n')
    tags_0[dicts_[tags00]] = 1
    tags0.append(tags_0)

batch_size = len(tags0)
z_dimension = 100
#z_batch0 = np.random.random([batch_size, z_dimension]) * 2 - 1 # for test images
z_batch0 = np.random.normal(size=[batch_size, z_dimension]) # for test images

## model initiate
gan_model = image_gan(batch_size)
gan_model.build_model()
saver = tf.train.Saver()

checkpoint_dir = './models/'

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)


    testImage = gan_model.generate_image(sess, z_batch0, tags0)
    testImage = testImage.reshape([batch_size, 64, 64, 3])
    testImage = (testImage + 1) / 2.
    for jj, im in enumerate(testImage):
        plt.subplot(math.ceil(len(tags0)/5), 5, jj+1)
        plt.axis('off')
        plt.imshow(im)

plt.show()

#plt.savefig('../../../gan-baseline/output_cgan.png')
