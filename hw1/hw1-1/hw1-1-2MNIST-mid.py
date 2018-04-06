from __future__ import print_function
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

# Training settings
parser = argparse.ArgumentParser(description='Pytorch MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 60000)')

parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='number of batch size for testing (default: 1000)')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.1, metavar='M',
                    help='SGD momentum (default: 0.1)')

parser.add_argument('--no-cuda', action='store_true', default=False, 
                    help='disable CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../../data', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                          ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,5)
        # an affine operation: y = Wx + b
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # If the size is a square you can only specify a single number
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net_deep(nn.Module):

    def __init__(self):
        super(Net_deep, self).__init__()
        self.conv1 = nn.Conv2d(1,10,5)
        self.conv2 = nn.Conv2d(10,20,5)
        self.fc1 = nn.Linear(320, 18)
        self.fc2 = nn.Linear(18, 18)
        self.fc3 = nn.Linear(18, 10)
        self.fc4 = nn.Linear(10,10)
        self.fc5 = nn.Linear(10,10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x)) 
        x = F.relu(self.fc4(x)) 
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)

class Net_s(nn.Module):

    def __init__(self):
        super(Net_s, self).__init__()
        self.conv1 = nn.Conv2d(1,4,5)
        self.fc1 = nn.Linear(576, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = x.view(-1,576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class Net_DNN_shallow(nn.Module):

    def __init__(self):
        super(Net_DNN, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10,10)
    def forward(self, x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc7(x)
        return F.log_softmax(x, dim=1)

class Net_DNN_mid(nn.Module):

    def __init__(self):
        super(Net_DNN, self).__init__()
        self.fc1 = nn.Linear(784, 18)
        self.fc2 = nn.Linear(18, 26)
        self.fc3 = nn.Linear(26,24)
        self.fc4 = nn.Linear(24,22)
        self.fc5 = nn.Linear(22,10)
        

    def forward(self, x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.log_softmax(x, dim=1)


class Net_DNN_deep(nn.Module):

    def __init__(self):
        super(Net_DNN, self).__init__()
        self.fc1 = nn.Linear(784, 17)
        self.fc2 = nn.Linear(17, 28)
        self.fc3 = nn.Linear(28,26)
        self.fc4 = nn.Linear(26,22)
        self.fc5 = nn.Linear(22,18)
        self.fc6 = nn.Linear(18,16)
        self.fc7 = nn.Linear(16,10)
        

    def forward(self, x):
        x = x.view(-1,784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return F.log_softmax(x, dim=1)



def train(epoch):
    L = []
    g_save = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        L.append(loss.data[0])
        grad_all = 0.0
        for p in model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy() ** 2).sum()
            grad_all = grad
        grad_norm = grad_all ** 0.5
        g_save.append(grad_norm)

        if batch_idx & args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss(g_norm): {:.6f}'.format(
                 epoch, batch_idx * len(data), len(train_loader.dataset),
                 100. * batch_idx / len(train_loader), loss.data[0]))
            
    return L, g_save

def test():
    model.eval()
    test_loss = 0
    acc_save = []
    loss_all = []
    correct = 0
    for data, target in train_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sun up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
         test_loss, correct, len(train_loader.dataset),
         100. * correct / len(train_loader.dataset)))

    acc_save.append(correct/len(train_loader.dataset))
    loss_all.append(test_loss)
    acc_save = np.array(acc_save)
    loss_all = np.array(loss_all)
    return acc_save, loss_all


def get_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


model = Net_DNN_mid()
print(model)
if args.cuda:
    model.cuda()
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.LBFGS(model.parameters())
pp = get_params(model)
print(pp)

LL = np.empty(0)
gg = np.empty(0)
ACC = np.empty(0)
l_a = np.empty(0)

for epoch in range(1, args.epochs + 1):
    
    L,g = train(epoch)
    acc, loss_test = test()
    print(acc)
    print(loss_test)
    L = np.array(L)
    g = np.array(g)
    gg = np.concatenate((gg, g), axis=0)
    #print('g = ',g.shape)
    LL = np.concatenate((LL,L), axis=0)
    ACC = np.concatenate((ACC,acc),axis=0)
    l_a = np.concatenate((l_a, loss_test), axis=0)
    #torch.save(model,'../../model_save/model_DNN_1_%s.pt' %epoch)
    if epoch % 3 ==0:
        print(epoch)
#        torch.save(model,'../../model_save/model_DNN_1_%s.pt' %epoch)
#np.save('./hw1_plot/gradient_norm', gg)
np.save('./npy_mnist/acc_DNN2', ACC)
np.save('./npy_mnist/loss_all_DNN2', l_a)
np.save('./npy_mnist/loss_DNN2', LL)
