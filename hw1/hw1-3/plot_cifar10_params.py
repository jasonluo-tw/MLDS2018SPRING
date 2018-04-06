import matplotlib.pyplot as plt
import numpy as np

datas = np.genfromtxt('./output_csv/cifar_params.csv', delimiter=',')

xx = datas[:, 0]
## train_loss, train_acc, test_loss, test_acc

plt.subplot(121)

plt.scatter(xx, datas[:,1], label='train loss')
plt.scatter(xx, datas[:,3], label='test loss')
plt.xlabel('parameters')
plt.ylabel('cross entropy')
plt.legend(loc='best')

plt.subplot(122)

plt.scatter(xx, datas[:,2], label='train acc')
plt.scatter(xx, datas[:,4], label='test acc')
plt.xlabel('parameters')
plt.ylabel('Accuracy')
plt.legend(loc='best')

plt.show()
