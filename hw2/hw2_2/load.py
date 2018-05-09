import torch 
import re
import os
import unicodedata
from gensim.models import Word2Vec
from config import MAX_LENGTH, save_dir

SOS_token = 0
EOS_token = 1
PAD_token = 2
UNK_token = 3

prefiex = ("我", "我們", "你", "你們" ,"妳" ,"妳們" ,"他" ,"他們" ,"她" ,"她們" ,"它" ,"它們")

class Voc:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "BOS", 1: "EOS", 2: "PAD", 3: "UNK"}
        self.n_words = 4

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else: 
            self.word2count += 1

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s 

def readVocs(corpus, corpus_name):
    print("Reading lines.....")

    # combine every two lines into pairs and normalize\
    with open(corpus) as f:
        content = f.read()
    parts = content.split("+++$+++")
    parts = [part.strip().split('\n') for part in parts]
    pairs = []
    for part in parts:
        for i in range(len(part) - 1):
            pairs.append(["".join(part[i].split()), "".join(part[i+1].split())])

    voc = Voc(corpus_name)
    print(pairs[0:10])
    return voc, pairs

def filterPair(p):
    return len(p[0]) < MAX_LENGTH and \
            len(p[1]) < MAX_LENGTH and \
            p[0].startswith(prefiex)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(corpus, corpus_name, word_model):
    voc, pairs = readVocs(corpus, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    #for pair in pairs:
    #    voc.addSentence(pair[0])
    #    voc.addSentence(pair[1])
    for i in range(len(word_model.wv.vocab)):
        voc.addWord(word_model.wv.index2word[i])

    print("Counted words:", voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs


def prepareData_test(corpus, corpus_name, word_model):
    voc, pairs = readVocs(corpus, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    #pairs = filterPairs(pairs)
    #print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for i in range(len(word_model.wv.vocab)):
        voc.addWord(word_model.wv.index2word[i])

    print("Counted words:", voc.n_words)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(voc, os.path.join(directory, '{!s}.tar'.format('voc')))
    torch.save(pairs, os.path.join(directory, '{!s}.tar'.format('pairs')))
    return voc, pairs

def loadPrepareData_test(corpus):
    word_model = Word2Vec.load('./save/training_data/conversation/word_emb')
    print(corpus)
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData_test(corpus, corpus_name, word_model)
    return voc, pairs

def loadPrepareData(corpus):
    word_model = Word2Vec.load('./save/training_data/conversation/word_emb')
    corpus_name = corpus.split('/')[-1].split('.')[0]
    try:
        print("Start loading training data ...")
        voc = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'voc.tar'))
        pairs = torch.load(os.path.join(save_dir, 'training_data', corpus_name, 'pairs.tar'))
    except FileNotFoundError:
        print("Saved data not found, start preparing trianing data ...")
        voc, pairs = prepareData(corpus, corpus_name, word_model)
    return voc, pairs

def gensim_model(corpus, corpus_name, SIZE=128, MIN_COUNT=8):
    with open(corpus) as f:
        content = f.read()
    parts = content.split("+++$+++")
    parts = [part.strip().split('\n') for part in parts]
    sentences = []
    for part in parts:
        for i in range(len(part)):
            sentences.append("".join(part[i].split()))

    model = Word2Vec(sentences, size=SIZE, window=5, min_count=MIN_COUNT, workers=8)
    directory = os.path.join(save_dir, 'training_data', corpus_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(os.path.join(directory, '{!s}'.format('word_emb')))
    return model, sentences


#SIZE = 128
#MIN_COUNT = 200
#word_model, sentences = gensim_model("./conversation.txt", 'conversation', SIZE, MIN_COUNT)
#voc, pairs = prepareData("./conversation.txt", 'movie_chat', word_model)
#print(pairs[0:10])
