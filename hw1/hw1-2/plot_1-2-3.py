import numpy as np
import matplotlib.pyplot as plt

#g_norm = np.load('./gradient_norm.npy')
loss = np.load('./npy_data/loss_DNN3.npy')

min_ratin = np.load('./npy_data/min_ratio.npy')
plt.xlabel("min_ratio")
plt.ylabel("loss")
plt.scatter(min_ratin, loss)
plt.savefig('./img/1-2-3.png')
plt.show()
