import matplotlib.pyplot as plt
import numpy as np

datas = np.genfromtxt('./output_csv/random_loss_acc.csv', delimiter=',')

plt.plot(datas[:,0], label='train loss')
plt.plot(datas[:,1], label='test loss')
plt.legend(loc='best')
plt.xlabel('epoch')
plt.ylabel('cross entropy')
plt.show()
