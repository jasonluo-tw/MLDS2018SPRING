import numpy as np
import math
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision.transforms as transfroms
from torch.autograd import Variable

class shallow_net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,230)
        self.out = nn.Linear(230,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x 


class middle_net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,16)
        self.fc2 = nn.Linear(16,12)
        self.fc3 = nn.Linear(12,13)
        self.fc4 = nn.Linear(13,19)
        self.out = nn.Linear(19,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x 

class deep_net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc4 = nn.Linear(10,10)
        self.fc5 = nn.Linear(10,10)
        self.fc6 = nn.Linear(10,10)
        self.fc7 = nn.Linear(10,10)
        self.out = nn.Linear(10,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc7(x))
        x = self.out(x)
        return x


## main program

model = nn_model()
xx = np.linspace(0.01,1,400000)
yy_ = np.sin(5*np.pi*xx) / (5*np.pi*xx)

## shuffle data
indices = np.arange(xx.shape[0])
np.random.shuffle(indices)
train = xx[indices]
label = yy_[indices]
# split data to train and val

nb_validation_samples = int(0.1 * train.shape[0])
train_ = train[nb_validation_samples:]
train_val = train[0:nb_validation_samples]


label_ = label[nb_validation_samples:]
label_val = label[0:nb_validation_samples]

## Normalizaion
#train_, train_val = normalize(train_, train_val)

model = shallow_net()
if torch.cuda.is_available():  # use GPU
    model.cuda()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(train_):
        if torch.cuda.is_available():
            data = Variable(data.view(-1,1).cuda())
        else:
            data = Variable(data.view(-1,1))
        # Forward pass only get output
        outputs = model(data)
        
        total += data.size(0)

        if torch.cuda.is_available():
            correct += (
