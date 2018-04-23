import tensorflow as tf
import numpy as np

## test input
test11 = np.load('./MLDS_hw2_1_data/training_data/feat/-AwoiGR6c8M_10_14.avi.npy')
test11 = np.reshape(test11, (1, 80, 4096))

num_units = 128
#with tf.variable_scope('encoder'):
video_input = tf.placeholder(tf.float32, [None, 80, 4096])
decoder_inputs_train = tf.placeholder(tf.float32, shape=(None, None), name='decoder_inputs')
## video_input: [batch_size, 80, 4096]
## Build RNN cell for two layers
rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]

## stack up RNN cell
multi_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

## Run Dynamic RNN
##  encoder_outputs: [max_time, batch_size, num_units]
##  encoder_state: [batch_size, num_units]
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(multi_encoder_cell, video_input, dtype=tf.float32)

### Done encoder 

## decoder cell
decoder_layer = [tf.nn.rnn_cell.LSTMCell(size) for size in [128, 256]]
multi_decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_layer)
## deocder embeddings
decoder_embeddings = tf.get_variable(name='embedding', 
        shape=[num_decoder_symbols, embedding_size])

decoder_emb_input = tf.nn.embedding_lookup(params = decoder_embeddings, ids = decoder_inputs_train)
# Helper
helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, name='training_helper')

# Decoder
#fc_1 = tf.layers.dense(logits, 512, activation=tf.nn.relu)
projection_layer = tf.layers.dense(dicts_size, activation=tf.nn.softmax)

decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell, helper, encoder_state, output_layer=projection_layer)

## Dynamic decoder
outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)   ## need to change
logits = outputs.rnn_output

## define loss cross_entropy
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_labels, logits=logits) ## need to change
train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)

## Calculate and clip gradients
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
max_gradient_morm = 1  ## can change
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

# Optimization
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer()  # can change learning rate
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

init = tf.global_variables_initializer()
## sess
with tf.Session() as sess:
    sess.run(init)
    aa = sess.run(encoder_outputs, feed_dict={video_input: test11})
    print(aa)
