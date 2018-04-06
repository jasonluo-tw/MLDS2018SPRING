import numpy as np
import matplotlib.pyplot as plt

norm = np.load('./npy_data/gradient_norm.npy')
loss = np.load('./npy_data/loss.npy')
norm = np.concatenate((np.array([0]), norm),axis=0)
x = np.linspace(1,norm.shape[0],num=norm.shape[0])

plt.figure(1,figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(x[0:17500], norm[0:17500], label='grad')
plt.ylabel('grad')
plt.subplot(2,1,2)
plt.plot(x[0:17500], loss[0:17500])
plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()
plt.savefig('./img/1-2-2.png')
