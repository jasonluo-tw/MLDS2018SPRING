import matplotlib.pyplot as plt
import numpy as np
f = open('./models2/training_scores_test_seed.txt', 'r')

datas = f.readlines()
cc = []
for data in datas:
    cc.append(float(data.split(':')[1]))

mean = []
for i in range(len(cc)-30):
    mean.append(np.mean(cc[i:(i+30)]))

mean = np.array(mean)
plt.plot(cc, color='red', alpha=0.5)
plt.plot(mean, 'red')
plt.plot([1, len(cc)], [3, 3], 'k')
print(len(mean[np.where(mean > 3)]) / len(mean))
print(len(mean))
print('Max:', np.max(mean))
plt.show()
