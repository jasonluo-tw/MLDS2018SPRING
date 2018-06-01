import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cgan import image_gan

print("Generate images")
batch_size = 25
z_dimension = 100
#z_batch0 = np.random.random([batch_size, z_dimension]) * 2 - 1 # for test images
z_batch0 = np.random.normal(size=[batch_size, z_dimension]) # for test images
tags00 = [0] *52 + [1] + [0] * 67
tags0 = [tags00 for i in range(batch_size)]
### init the model
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
        plt.subplot(5, 5, jj+1)
        plt.axis('off')
        plt.imshow(im)

#plt.show()
plt.savefig('../../../gan-baseline/output_cgan.png')
