import numpy as np
from matplotlib import pyplot as plt
import itertools
test1 = np.load('./pca_data/pca1.npy')
test2 = np.load('./pca_data/pca2.npy')
test3 = np.load('./pca_data/pca3.npy')
test4 = np.load('./pca_data/pca4.npy')
test5 = np.load('./pca_data/pca5.npy')
test6 = np.load('./pca_data/pca6.npy')
test7 = np.load('./pca_data/pca7.npy')
test8 = np.load('./pca_data/pca8.npy')
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
plt.title('Full_MODEL')
plt.savefig('./img/pca_all.png')

plt.show()
