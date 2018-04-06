import numpy as np
import matplotlib.pyplot as plt

#g_norm = np.load('./gradient_norm.npy')
loss = np.load('./loss_DNN3.npy')

min_ratin = np.load('./min_ratio.npy')
plt.xlabel("min_ratio")
plt.ylabel("loss")
plt.scatter(min_ratin, loss)
plt.savefig('./1-2-3.png')
plt.show()
