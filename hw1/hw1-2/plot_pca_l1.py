import numpy as np
from matplotlib import pyplot as plt
import itertools
test1 = np.load('./pca_data/pca_l11.npy')
test2 = np.load('./pca_data/pca_l12.npy')
test3 = np.load('./pca_data/pca_l13.npy')
test4 = np.load('./pca_data/pca_l14.npy')
test5 = np.load('./pca_data/pca_l15.npy')
test6 = np.load('./pca_data/pca_l16.npy')
test7 = np.load('./pca_data/pca_l17.npy')
test8 = np.load('./pca_data/pca_l18.npy')
data = [test1, test2, test3, test4, test5, test6, test7, test8]
colors = ["r", "b","g","y","navy","m","crimson","grey"]
plt.figure(1, figsize=(10,10))
txt = np.arange(0,100,3)
txt = txt[1::]
print(txt.shape)
j = 0
for ha in data:
    print(ha.shape)
    for i in range(ha.shape[0]):
        plt.scatter(ha[i,0], ha[i,1],c=colors[j])
    j+=1
plt.title('layer1')

plt.savefig('./img/pca_layer1.png')

plt.show()
