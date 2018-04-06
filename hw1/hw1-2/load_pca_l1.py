from __future__ import print_function
from sklearn.decomposition import PCA as PCA 
import argparse
import torch
import torchvision
from torchvision import datasets ,transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
import os 
from sklearn.preprocessing import StandardScaler
import re

def atof(text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

def natural_keys(text):
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


class Net_DNN(nn.Module):

    def __init__(self):
        super(Net_DNN, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10,10)

    def forward(self, x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


ddd =os.listdir('./pca_data/')
ddd.sort(key=natural_keys)


for m in range(1,9):

    test1 = np.empty((1,15680))
    for d in ddd:
        if 'model_DNN_'+ str(m) +'_' in d:
            model = torch.load('./pca_data/'+d)
            params = model.state_dict()
            epoch3 = params['fc1.weight'].cpu().numpy().flatten()
            epoch3 = epoch3.reshape(1,-1)
            print(epoch3.shape)
            test1 = np.concatenate((test1, epoch3), axis=0)
    #sc = StandardScaler()
    #test1 = sc.fit_transform(test1[1::,:])
    print(test1.shape)
    pca = PCA(n_components=2)
    new = pca.fit_transform(test1[1::,:])
    print(test1.shape)
    np.save('./pca_data/pca_l1'+str(m)+'.npy', new)


