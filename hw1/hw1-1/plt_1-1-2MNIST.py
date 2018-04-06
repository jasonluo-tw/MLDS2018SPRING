import numpy as np
from matplotlib import pyplot as plt

loss_shallow = np.load('./npy_mnist/loss_DNN1.npy')
loss_shallow = loss_shallow[::938]
loss_1 = np.load('./npy_mnist/loss_all_DNN1.npy')
loss_middle = np.load('./npy_mnist/loss_DNN2.npy')
loss_middle = loss_middle[::938]
loss_2 = np.load('./npy_mnist/loss_all_DNN2.npy')
loss_deep = np.load('./npy_mnist/loss_DNN3.npy')
loss_deep = loss_deep[::938]
loss_3 = np.load('./npy_mnist/loss_all_DNN3.npy')

acc_shallow = np.load('./npy_mnist/acc_DNN1.npy')
acc_middle = np.load('./npy_mnist/acc_DNN2.npy')  
acc_deep = np.load('./npy_mnist/acc_DNN3.npy')

x = np.linspace(0, 100, 100)
plt.figure(1, figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(x,loss_shallow, label='shallow')
plt.plot(x,loss_middle, label='middle')
plt.plot(x, loss_deep, label='deep')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(2,1,2)
plt.plot(x,loss_1, label='shallow')
plt.plot(x,loss_2, label='middle')
plt.plot(x,loss_3, label='deep')

plt.savefig('./pics/1-1-2loss.png')

plt.figure(2, figsize=(12,6))
plt.plot(x, acc_shallow, label='acc_shollow')
plt.plot(x, acc_middle, label='acc_middle')
plt.plot(x, acc_deep, label='acc_middle')

plt.legend()
plt.savefig('./pics/1-1-2acc.png')
plt.show()
