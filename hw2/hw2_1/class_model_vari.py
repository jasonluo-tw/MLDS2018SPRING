import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.python.layers import core as layers_core

class video2seq():
    """ video seq2seq model
    Argv:
     necessary input
      dicts_size: dictionary size
      max_length: max label seq length. This length is the same as decoder length
      nn: batch size

     not necessary
      necessary beam_width = 1 for training, if use beam search can change width
      embed_size: embedding vector size, default=64
      is_inference: use for training or inference
    """
    def __init__(self, dicts_size, max_length, nn, samp_pro=0.6, beam_width=1,
            num_units=512, embed_size=64, is_inference=False):

        self.batch_size = nn
        self.dicts_size = dicts_size
        self.max_length = max_length
        self.num_units = num_units
        self.embed_size = embed_size
        self.is_inference = is_inference
        self.beam_width = beam_width
        self.samp_pro = samp_pro
    def init_placeholders(self):

        self.video_inputs = tf.placeholder(tf.float32, [None, 80, 4096], name='video_in')
        self.label_inputs = tf.placeholder(tf.int32, [None, self.max_length],
                name = 'true_labels')
        self.decoder_inputs_train = tf.placeholder(tf.int32, shape=(None, self.max_length), 
                name = 'train_de_inputs')
        self.target_weights = tf.placeholder(tf.float32, shape=(None, self.max_length),
                name = 'target_weights')
        ###### need to implemention in the future
        # decoder_lengths = tf.placeholder(tf.int32, shape=(batch_size), name='decoder_lengths')
        self.decoder_lengths = [self.max_length] * self.batch_size
    
    def build_model(self):
        print('building model ...')

        ## prepare init_placeholders
        self.init_placeholders()
        
        ## build encoder ##
        
    def input_data(self, video_in, label_in, decoder_in, target_weights, decoder_lengths=None):
        self.inputs = {}
        self.inputs[self.video_inputs] = video_in
        self.inputs[self.label_inputs] = label_in
        self.inputs[self.decoder_inputs_train] = decoder_in
        self.inputs[self.target_weights] = target_weights
    def input_test_data(self, video_in):
        self.inputs = {}
        self.inputs[self.video_inputs] = video_in

    def build_encoder(self):
        ## video_input: [batch_size, 80, 4096]
        #with scope
        print('build encoder ...')
        num = self.num_units
        rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [num*2, num, num, num]]

        ## stack up RNN cell
        multi_encoder_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

        ## Run Dynamic RNN
        ## encoder_outputs: [batch_size, max_time, num_units]
        ## encoder_state: [batch_size, num_units]
        encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                multi_encoder_cell, self.video_inputs, dtype=tf.float32)
        
        ## Attention mechanism (Luong)
        #self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.num_units, encoder_outputs)

        ## for training beam_width = 1, greedy testing beam_width =1, beam search = 1,2,3,4,5
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_width)
        self.attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.num_units * 2, tiled_encoder_outputs)


    def build_decoder(self):
        ## decoder cell
        ## with scope
        print('build decoder ...')
        top_cell0 = tf.nn.rnn_cell.LSTMCell(self.num_units*2, name='top_lstm_cell')
        top_cell = tf.contrib.seq2seq.AttentionWrapper(top_cell0, self.attention_mechanism,
                attention_layer_size=self.num_units*2)

        top_state = top_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state = self.encoder_state[0])
        ### use more than one LSTM unlock below
        #sec_cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
        #sec_state = sec_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
        #third_cell = tf.nn.rnn_cell.LSTMCell(self.num_units * 3)
        #third_state = third_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

        ## stack up decoder cell
        #multi_decoder_cell = tf.nn.rnn_cell.MultiRNNCell([top_cell, sec_cell])
        #decoder_init_state = (top_state, sec_state)
        ## decoder embeddings
        decoder_embeddings = tf.get_variable(name='embedding', 
                shape=[self.dicts_size, self.embed_size])
        decoder_emb_input = tf.nn.embedding_lookup(params = decoder_embeddings, ids = self.decoder_inputs_train)
        
        ## porjection layer
        projection_layer = layers_core.Dense(self.dicts_size, activation=tf.nn.relu, use_bias=True)

        if self.beam_width == 1:
            ## helper 
            ## training_helper
            #train_helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_input, 
            #        self.decoder_lengths, name='training_helper')
            train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(decoder_emb_input,
                    self.decoder_lengths, 
                    decoder_embeddings,
                    sampling_probability=self.samp_pro, name='scheduled_sampling')
            ## eval, test helper
            test_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings, tf.fill([self.batch_size], 1), 2) # 1 is start token, 2 is end token

            ## decoder stage
            ## train decoder
            self.train_decoder = tf.contrib.seq2seq.BasicDecoder(top_cell, train_helper, top_state, output_layer=projection_layer)
            ## test decoder
            self.test_decoder = tf.contrib.seq2seq.BasicDecoder(top_cell, test_helper, top_state, output_layer=projection_layer)
            
            ######### training decoder ########
            ## dynamic decoder
            outputs, _, _= tf.contrib.seq2seq.dynamic_decode(self.train_decoder, maximum_iterations=self.max_length)
            
            logits = outputs.rnn_output
            self.train_eval = tf.argmax(tf.nn.softmax(logits), 2)
            
            ## define loss cross_entropy
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_inputs, logits=logits)
            self.train_loss = tf.reduce_sum(crossent * self.target_weights / self.batch_size)
            ## calculate and clip gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(self.train_loss, params)
            max_gradient_norm = 1 ## can change
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

            ## optimization
            learning_rate = 0.001
            optimizer = tf.train.AdamOptimizer() ## can change learning rate
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, params))
            #############################

            ######## test decoder #######
            test_outputs, _, _= tf.contrib.seq2seq.dynamic_decode(self.test_decoder, maximum_iterations=self.max_length)
            test_logits = test_outputs.rnn_output
            ## get decode length
            test_logit_len = tf.shape(test_logits)[1]
            #test_logit_len = tf.Print(test_logit_len, [test_logit_len], message='test_len:')
            
            ## pad decode length
            pad_size = self.max_length - test_logit_len
            test_logits = tf.pad(test_logits, [[0, 0], [0, pad_size], [0, 0]], "CONSTANT")
            #self.test_logits_shape = tf.shape(test_logits)
            self.test_eval = tf.argmax(tf.nn.softmax(test_logits), 2)
            
            ## define loss cross_entropy
            crossent_test = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_inputs, logits=test_logits)
            self.test_loss = tf.reduce_sum(crossent_test * self.target_weights / self.batch_size)
        else:
            #### implement beamsearch 
            ## beam_width: 5
            ## Replicate encoder infos beam_width times
            #beam_cell = tf.contrib.seq2seq.AttentionWrapper(top_cell0, 
            #        self.attention_mechanism2, 
            #        attention_layer_size=self.num_units)
            beam_decoder_state = tf.contrib.seq2seq.tile_batch(self.encoder_state[0], multiplier=self.beam_width)
            beam_state = top_cell.zero_state(batch_size=self.batch_size * self.beam_width, dtype=tf.float32).clone(cell_state = beam_decoder_state)
            ## define a beam-search decoder --- notice here cell use top_cell
            beam_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=top_cell, embedding=decoder_embeddings,
                    start_tokens=tf.fill([self.batch_size], 1), end_token=2, initial_state=beam_state,
                    beam_width=self.beam_width, output_layer=projection_layer, length_penalty_weight=0.0)
            ## beam search decoding
            beam_output, _, _ = tf.contrib.seq2seq.dynamic_decode(beam_decoder, maximum_iterations=self.max_length)
            self.beam_predict = beam_output.predicted_ids

    def run_train(self, sess):
        sess.run(self.update_step, feed_dict=self.inputs)
        train_batchsize_loss = sess.run(self.train_loss, feed_dict=self.inputs)
        evaluation = sess.run(self.train_eval, feed_dict=self.inputs)

        return train_batchsize_loss, evaluation

    def run_eval(self, sess):
        test_batchsize_loss = sess.run(self.test_loss, feed_dict=self.inputs)
        evaluation = sess.run(self.test_eval, feed_dict=self.inputs)

        return test_batchsize_loss, evaluation

    def run_test(self, sess):
        return sess.run(self.test_eval, feed_dict=self.inputs)

    def run_beamsearch(self, sess):
        return sess.run(self.beam_predict, feed_dict=self.inputs)
