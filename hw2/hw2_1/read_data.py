import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from tqdm import tqdm

# import label data
label_datas = pd.read_json('./MLDS_hw2_1_data/training_label.json')
sentences = label_datas['caption'].tolist()

# start to prepare input train and output

w2vec = []
target_characters = []
max_seq_length = 0
max_cases = 0

for senti in sentences:
    for ss in senti:
        com = ['\n'] + ss.rstrip('.').split() + ['\t']
        max_cases += 1
        if len(com) > max_seq_length:
            max_seq_length = len(com)
        w2vec.append(com)

# set up word2vec model
w2vec_model = Word2Vec(w2vec, size=10, min_count=2)

labels = np.zeros((max_cases, max_seq_length), dtype='float32')

index_id = []
for i in range(len(label_datas)):
    idd = label_datas[i:(i+1)]['id'].values[0]
    captions = label_datas[i:(i+1)]['caption']
    for caption in captions:
        for ii, ss in enumerate(caption):
            index_id.append(idd)
            com = ['\n'] + ss.rstrip('.').split() + ['\t']
            for jj, word in enumerate(com):
                if word in w2vec_model.wv:
                    labels[ii, jj] = int(w2vec_model.wv.vocab[word].index + 1)
                else:
                    labels[ii, jj] = 0

## \n == 0+1 <BOS> \t == 1+1 <EOS>

#np.set_printoptions(suppress=True)
print(labels.shape[0])


# train data 1450*80*4096

#train_data = np.zeros((1450, 80, 4096), dtype='float32')
train_data = {}
trainlist = os.popen('ls ./MLDS_hw2_1_data/training_data/feat/').read().split()

# import training data
print('Start loading data...')
for index, fi in tqdm(enumerate(trainlist)):
    train_data[fi.rstrip('.npy')] = np.load('./MLDS_hw2_1_data/training_data/feat/' + fi)

#print(train_data[index_id[0]])
