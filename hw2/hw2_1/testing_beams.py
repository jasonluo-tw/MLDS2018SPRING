import numpy as np
import pandas as pd
from math import floor
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import tensorflow as tf
from class_model_vari import video2seq
#### load testing data
test_data = {}

testlist = os.popen('ls ./MLDS_hw2_1_data/testing_data/feat/').read().split()

## import testing data
print('Start loading testing data ...')
index_id = []
for index, fi in tqdm(enumerate(testlist)):
    index_id.append(fi.rstrip('.npy'))
    test_data[fi.rstrip('.npy')] = np.load('./MLDS_hw2_1_data/testing_data/feat/' + fi)

w2vec_model = Word2Vec.load('./dicts/w2vec.model')

dicts_size = len(w2vec_model.wv.vocab) + 1
max_seq_length = 27
batch_size = len(test_data)

beam_width = 5
model = video2seq(dicts_size, max_seq_length, batch_size, 0.6, beam_width)
model.build_model()
model.build_encoder()
model.build_decoder()

#checkpoint_dir = './models/bleu0469/'
checkpoint_dir = './models/pro06/'
#checkpoint_dir = './models/pro10/'

saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    saver.restore(sess, ckpt.model_checkpoint_path)

    video_in = np.array([test_data[ii] for ii in index_id])
    video_in.reshape(batch_size, 80, 4096)

    model.input_test_data(video_in)
    test_index = model.run_beamsearch(sess)
    result = []
    for ii in range(beam_width):
        result.append([])
        for sentence in test_index[:, :, ii]:
            cc = ''
            early_word = 'BOS'
            for indexx in sentence:
                if indexx == 2:
                    break
                else:
                    word_now = w2vec_model.wv.index2word[int(indexx-1)]
                    if word_now != early_word:
                        cc = cc + ''.join(word_now) + ' '
                        #print(early_word, '  ', word_now)
                        early_word = word_now
                    else:
                        pass
    
            result[ii].append(cc)

import csv

for ii, beam_result in enumerate(result):
    filename = './outputs/result7_beam'+str(ii)+'.txt'
    with open(filename, 'w+') as f:
        s = csv.writer(f, delimiter=',', lineterminator='\n')
        for i in range(len(index_id)):
            s.writerow([index_id[i], beam_result[i]])

print('Done')
