import tensorflow as tf
import numpy as np
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
from tensorflow.python.layers import core as layers_core


def model(embedding_size, dicts_size, max_label_length, nn, num_units=256):
    batch_size = nn

    #with tf.variable_scope('encoder'):
    video_inputs = tf.placeholder(tf.float32, [None, 80, 4096])
    label_inputs = tf.placeholder(tf.int32, [None, max_label_length], name='true_labels')
    decoder_inputs_train = tf.placeholder(tf.int32, shape=(None, max_label_length), name='decoder_inputs')
    target_weights = tf.placeholder(tf.float32, shape=(None, max_label_length), name='target_weights')
    #################### need to implemention in the future
    #decoder_lengths = tf.placeholder(tf.int32, shape=(batch_size), name='decoder_lengths')
    decoder_lengths = [max_label_length] * nn
    ####################
    ## video_input: [batch_size, 80, 4096]
    ## Build RNN cell for two layers
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [num_units, num_units * 2, num_units * 2]]
    
    ## stack up RNN cell
    multi_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    
    ## Run Dynamic RNN
    ##  encoder_outputs: [batch_size, max_time, num_units]
    ##  encoder_state: [batch_size, num_units]
    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(multi_encoder_cell, video_inputs, dtype=tf.float32)
    
    ## Attention mechanism (Luong)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, encoder_outputs)
    
    
    ### Done encoder 
    
    ## decoder cell
    top_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    top_cell = tf.contrib.seq2seq.AttentionWrapper(top_cell, attention_mechanism, attention_layer_size=num_units)
    top_state = top_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=encoder_state[0])
    
    sec_cell = tf.nn.rnn_cell.LSTMCell(num_units * 2)
    sec_state = sec_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    
    third_cell = tf.nn.rnn_cell.LSTMCell(num_units * 2)
    third_state = third_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

    multi_decoder_cell = tf.nn.rnn_cell.MultiRNNCell([top_cell, sec_cell])

    decoder_initial_state = (top_state, sec_state)
    ## deocder embeddings
    decoder_embeddings = tf.get_variable(name='embedding', 
            shape=[dicts_size, embedding_size])
    
    decoder_emb_input = tf.nn.embedding_lookup(params = decoder_embeddings, ids = decoder_inputs_train)
    # Helper
    helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_input, decoder_lengths, name='training_helper')
    
    # Decoder
    projection_layer = layers_core.Dense(dicts_size, activation=tf.nn.relu, use_bias=True) # neet to change
    
    decoder = tf.contrib.seq2seq.BasicDecoder(top_cell, helper, top_state, output_layer=projection_layer)
    
    ## Dynamic decoder
    outputs, _, _= tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_label_length)   ## need to change
    logits = outputs.rnn_output
    eval_ = tf.argmax(tf.nn.softmax(logits), 2)    
    ## define loss cross_entropy
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_inputs, logits=logits) ## need to change
    #train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)
    train_loss = tf.reduce_sum(crossent * target_weights / batch_size)
    
    ## Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(train_loss, params)
    max_gradient_norm = 1  ## can change
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
    
    # Optimization
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer()  # can change learning rate
    update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
    
    return update_step, train_loss, video_inputs, label_inputs, decoder_inputs_train, target_weights, eval_, decoder_lengths

if __name__ == '__main__':
    print('hello')
#init = tf.global_variables_initializer()
## sess
#with tf.Session() as sess:
#    sess.run(init)
