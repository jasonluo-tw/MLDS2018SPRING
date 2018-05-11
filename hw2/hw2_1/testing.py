import numpy as np
import pandas as pd
from math import floor
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import tensorflow as tf
from class_model_vari import video2seq
import sys
### prepare directory path
test_path = sys.argv[1].rstrip('/')
output_path = sys.argv[2].rstrip('/')
#### load testing data
test_data = {}

testlist = os.popen('ls '+test_path+'/feat/').read().split()

## import testing data
print('Start loading testing data ...')
index_id = []
for index, fi in tqdm(enumerate(testlist)):
    index_id.append(fi.rstrip('.npy'))
    test_data[fi.rstrip('.npy')] = np.load(test_path + '/feat/' + fi)

w2vec_model = Word2Vec.load('./dicts/w2vec.model')

dicts_size = len(w2vec_model.wv.vocab) + 1
max_seq_length = 27
batch_size = len(test_data)

model = video2seq(dicts_size, max_seq_length, batch_size, 0.6)
model.build_model()
model.build_encoder()
model.build_decoder()

checkpoint_dir = './models/'

saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    saver.restore(sess, ckpt.model_checkpoint_path)

    video_in = np.array([test_data[ii] for ii in index_id])
    video_in.reshape(batch_size, 80, 4096)

    model.input_test_data(video_in)
    test_index = model.run_test(sess)
    result = []
    for sentence in test_index:
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

        result.append(cc)

import csv
out = []
for i in range(len(index_id)):
    out.append([index_id[i], result[i]])

filename = output_path
with open(filename, 'w+') as f:
    s = csv.writer(f, delimiter=',', lineterminator='\n')
    for i in range(len(out)):
        s.writerow(out[i])

print('Done')
