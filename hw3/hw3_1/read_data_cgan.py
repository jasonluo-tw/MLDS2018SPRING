import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from random import randint

def setup_dicts(dirs):
    with open(dirs, 'r') as f:
        datas = f.readlines()

    hair_ = []
    eye_ = []
    all_dicts = {}
    dict_index = 0
    for index, data in enumerate(datas):
        feature0 = data.split(',')[-1].split()
        hair = feature0[0] + ' ' + feature0[1]
        eye = feature0[2] + ' ' + feature0[3]
        if hair not in hair_:
            hair_.append(hair)
        if eye not in eye_:
            eye_.append(eye)

    for i in hair_:
        for j in eye_:
            ff = i + ' ' + j
            if ff not in all_dicts:
                all_dicts[ff] = dict_index
                dict_index += 1

    return all_dicts


class read_imgs():
    def __init__(self):
        self.length = 0
        self.now_size = 0
        self.imgs = []
        self.fake = False
        self.tags = False

    def read_source1(self, dirs):

        from tqdm import tqdm
        print("read data from: ", dirs)
        trainlist = os.popen('ls '+dirs).read().split()

        for index in tqdm(range(len(trainlist))):
            name = str(index) + '.jpg'
            self.imgs.append(plt.imread(dirs+name))


        ## define some variables
        self.length = self.length + len(trainlist)

    def read_source1_tags(self, dirs, dicts, fake=True):
        
        # define some variables
        dicts_length = len(dicts)
        self.fake = fake
        self.tags = True
        
        with open(dirs, 'r') as f:
            datas = f.readlines()

        self.features = []
        if fake:
            self.fake_features = []
        
        # start to make one-hot encoding
        for index, data in enumerate(datas):
            self.features.append(np.zeros(dicts_length))
            feature0 = data.split(',')[-1].rstrip('\n')
            
            
            if fake:
                self.fake_tags_g(index, feature0, dicts, dicts_length) 

            self.features[index][dicts[feature0]] = 1  # one-hot encoding

    def fake_tags_g(self, index, feature0, t2i_dicts, dicts_length):
        self.fake_features.append(np.zeros(dicts_length))
        cc = int(dicts_length/2)
        
        aa = randint(1, cc-1)
        
        if t2i_dicts[feature0] <= cc:
            self.fake_features[index][t2i_dicts[feature0] + aa] = 1
        else:
            self.fake_features[index][t2i_dicts[feature0] - aa] = 1
        

    def next_batch(self, batch_size, normal=False): 
        
        next_size = self.now_size + batch_size
        if next_size > self.length:
            next_imgs0 = np.array(self.imgs[self.now_size:])
            
            next_size = next_size - self.length
            next_imgs1 = np.array(self.imgs[0:next_size])

            next_imgs = np.concatenate((next_imgs0, next_imgs1), axis=0)
            
            if self.tags:
                tags0 = self.features[self.now_size:]
                tags1 = self.features[0:next_size]

                next_tags = np.concatenate((tags0, tags1), axis=0)
                
                # generate fake texts
                if self.fake:
                    fake0 = self.fake_features[self.now_size:]
                    fake1 = self.fake_features[0:next_size]
                    next_fakes = np.concatenate((fake0, fake1), axis=0)

        elif next_size == self.length:
            next_imgs = np.array(self.imgs[self.now_size:])

            if self.tags:
                next_tags = self.features[self.now_size:]

                # generate fake texts
                if self.fake:
                    next_fakes = self.fake_features[self.now_size:]
            
            next_size = 0

        else:
            next_imgs = np.array(self.imgs[self.now_size:next_size])

            if self.tags:
                next_tags = self.features[self.now_size:next_size]
                
                # generate fake texts
                if self.fake:
                    next_fakes = self.fake_features[self.now_size:next_size]
        
        self.now_size = next_size
        
        # if normal=True, normalize data from -1 to 1
        if normal:
            next_imgs = (next_imgs / 255.) * 2 - 1

        if self.tags:    
            if self.fake:
                return next_imgs, next_tags, next_fakes
            else:
                return next_imgs, next_tags

        else:
            return next_imgs

    def read_source2(self, dirs, resize = False):
        
        from tqdm import tqdm
        print("read data from: ", dirs)
        trainlist = os.popen('ls '+dirs).read().split()
        imgs = []

        for index, name in tqdm(enumerate(trainlist)):
            if resize:
                img0 = Image.open(dirs+name)
                img0.thumbnail((64, 64), Image.ANTIALIAS)
                self.imgs.append(np.array(img0))
            else:
                self.imgs.append(plt.imread(dirs+name))
        
        # define
        self.length = self.length + len(trainlist)
    
    def shuffle_data(self):
        import random
        random.Random(5).shuffle(self.imgs)
        
        if self.tags:
            random.Random(5).shuffle(self.features)
            if self.fake:
                random.Random(5).shuffle(self.fake_features)

if __name__ == '__main__':

    #datasets = read_imgs()
    #datasets.read_source1('./dataset/extra_data/images/')
    dicts_ = setup_dicts('./dataset/extra_data/tags.csv')
    #datasets.read_source1_tags('./dataset/extra_data/tags.csv', dicts_) 
    #datasets.shuffle_data()
    #next_imgs, next_tags, next_fake = datasets.next_batch(200)
    print(dicts_)
    #print(next_tags)
    #print(next_fake)

