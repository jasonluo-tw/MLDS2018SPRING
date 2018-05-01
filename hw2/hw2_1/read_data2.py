import numpy as np
import pandas as pd
from math import floor
import os
from gensim.models import Word2Vec
from tqdm import tqdm
import model2
import tensorflow as tf
# import label data
label_datas = pd.read_json('./MLDS_hw2_1_data/training_label.json')
sentences = label_datas['caption'].tolist()

# start to prepare input train and output

w2vec = []
target_characters = []
max_seq_length = 0
max_cases = 0

for senti in sentences:
    check_ = 0
    for ss in senti:
        com = ss.rstrip('.').split()
        if len(com) > 20:
            continue
        if check_ == 1:
            break
        check_ = 1
        com = ['<BOS>'] + ss.rstrip('.').split() + ['<EOS>'] + ['<UNK>']
        max_cases += 1
        if len(com) > max_seq_length:
            max_seq_length = len(com)
        w2vec.append(com)

# set up word2vec model
w2vec_model = Word2Vec(w2vec, size=1, min_count=2)
del w2vec
del sentences


labels = np.zeros((max_cases, max_seq_length), dtype='float32')

decoder_inputs = np.zeros((max_cases, max_seq_length), dtype='float32')
decoder_length = np.zeros((max_cases), dtype='float32')

loss_mask = np.zeros((max_cases, max_seq_length), dtype='float32')

## test

ii = 0 
index_id = []
compare = []
for i in range(len(label_datas)):
    idd = label_datas[i:(i+1)]['id'].values[0]
    captions = label_datas[i:(i+1)]['caption'].values
    for caption in captions:
        check_ = 0
        for ss in caption:
            com = ss.rstrip('.').split()
            if len(com) > 20:
                continue
            if check_ == 1:
                break
            check_ = 1
            index_id.append(idd)
            decoder_inputs[ii, 0] = (w2vec_model.wv.vocab['<BOS>'].index + 1)
            compare.append(ss)
            for jj, word in enumerate(com):
                if word in w2vec_model.wv:
                    labels[ii, jj] = int(w2vec_model.wv.vocab[word].index + 1)
                    decoder_inputs[ii, jj + 1] = int(w2vec_model.wv.vocab[word].index + 1)
                else:
                    labels[ii, jj] = int(w2vec_model.wv.vocab['<UNK>'].index + 1)
                    decoder_inputs[ii, jj + 1] = int(w2vec_model.wv.vocab['<UNK>'].index + 1)

            labels[ii, len(com)] = int(w2vec_model.wv.vocab['<EOS>'].index + 1)
            decoder_length[ii] = len(com) + 1
            loss_mask[ii, 0:(len(com)+1)] = 1

            ii += 1

##  <BOS>=0+1=1   <EOS>=1+1=2  <UNK>=2+1=3 PAD=0
#np.set_printoptions(suppress=True)

dicts_size = len(w2vec_model.wv.vocab) + 1
print('<BOS> :', w2vec_model.wv.vocab['<BOS>'].index, 'model looks at 1')
print('<EOS> :', w2vec_model.wv.vocab['<EOS>'].index, 'model looks at 2')
print('<UNK> :', w2vec_model.wv.vocab['<UNK>'].index, 'model looks at 3')
#print(max_seq_length)
#print(len(index_id), len(decoder_inputs), len(labels), len(loss_mask), len(decoder_length))

# train data 1450*80*4096

#train_data(1450, 80, 4096)
train_data = {}
trainlist = os.popen('ls ./MLDS_hw2_1_data/training_data/feat/').read().split()

# import training data
print('Start loading data...')
for index, fi in tqdm(enumerate(trainlist)):
    train_data[fi.rstrip('.npy')] = np.load('./MLDS_hw2_1_data/training_data/feat/' + fi)

#print(train_data[index_id[0]])



## define the batch size epochs
epochs = 100
batch_size = 50

update_step, train_loss, video_inputs, label_inputs, decoder_inputs_train, target_weights, eval_, _ = model2.model(64, dicts_size, max_seq_length, batch_size)

init = tf.global_variables_initializer()

## shuffle data
index_id = np.array(index_id)
shuffle_indices = np.arange(index_id.shape[0])
np.random.shuffle(shuffle_indices)
index_id = index_id[shuffle_indices]
labels = labels[shuffle_indices]
decoder_inputs = decoder_inputs[shuffle_indices]
decoder_length = decoder_length[shuffle_indices]
loss_mask = loss_mask[shuffle_indices]

step_num = int(floor(len(labels) / batch_size))

## test data is right or not
compare = np.array(compare)
compare = compare[shuffle_indices]
#print(compare[3],'\n', index_id[3],'\n', labels[3],'\n',decoder_inputs[3],'\n',decoder_length[3],'\n', loss_mask[3])


## sess
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        # Train with batch
        for idx in range(1):
            indice = np.arange(idx*batch_size, (idx+1)*batch_size)
            #video_in = train_data[index_id[0]].reshape(batch_size, 80, 4096)
            video_in = np.array([train_data[ii] for ii in index_id[idx*batch_size:(idx+1)*batch_size]])
            video_in = video_in.reshape(batch_size, 80, 4096)
            label_ = labels[indice].reshape(batch_size, max_seq_length)
            decoder_ = decoder_inputs[indice].reshape(batch_size, max_seq_length)
            target_ = loss_mask[indice]
            #dec_len_ = decoder_length[indice]  # need to change in the future
            feed_input = {video_inputs: video_in, 
                          label_inputs: label_, 
                          decoder_inputs_train: decoder_,
                          target_weights: target_}
                          #decoder_lengths: dec_len_}
            
            sess.run(update_step, feed_dict=feed_input)
            
            if i % 5 == 0 and idx == 0:
                cross_loss = sess.run(train_loss, feed_dict=feed_input)
                eva_index = sess.run(eval_, feed_dict=feed_input)
                print('\n')
                print('now epochs:',i,'loss:', cross_loss)
                print('###################################')
                for indexx in eva_index[0]:
                    if indexx == 2:
                        break
                    else:
                        print(w2vec_model.wv.index2word[indexx-1], end=' ')
                print('\n')
                print('Ground truth:')
                for indexx in label_[0]:
                    if indexx == 2:
                        break
                    print(w2vec_model.wv.index2word[int(indexx-1)], end=' ')
    
