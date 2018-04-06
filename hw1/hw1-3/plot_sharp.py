import matplotlib.pyplot as plt
import numpy as np

with open('./output_csv/loss_acc_sharpness.csv', 'r') as f:
    datas = f.readlines()

    batch_size = np.zeros(len(datas))
    loss_train = np.zeros(len(datas))
    loss = np.zeros(len(datas))
    acc_train = np.zeros(len(datas))
    acc = np.zeros(len(datas))
    sharp = np.zeros(len(datas))

for i, data in enumerate(datas):
    batch_size[i], loss_train[i], loss[i], acc_train[i], acc[i], sharp[i] = list(map(float,data.rstrip('\n').split(',')))

for i in range(len(sharp)):
    sharp[i] = sharp[i] * 1e+5

fig = plt.figure(1)
ax = fig.add_subplot(211)

ax2 = ax.twinx()

ax.plot(batch_size, loss_train, 'orange', label='train loss')
ax.plot(batch_size, loss, 'orange', linestyle='--', label='test loss')


ax2.plot(batch_size, sharp, 'g-', label='sharpness')
fig.legend(loc=2, bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

ax.set_xlabel('batch_size')
ax.set_ylabel('cross entropy')

ax2.set_ylabel('sharpness')
plt.draw()
###

ax3 = fig.add_subplot(212)
ax4 = ax3.twinx()

ax3.plot(batch_size, acc_train, 'orange', label='train acc')
ax3.plot(batch_size, acc, 'orange', linestyle='--', label='test acc')

ax4.plot(batch_size, sharp, 'g-', label='sharpness')

h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax4.get_legend_handles_labels()

leg = ax4.legend(handles=h1+h2, labels=l1+l2, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax3.transAxes)
leg.get_frame().set_alpha(0.8)

ax3.set_xlabel('batch_size')
ax3.set_ylabel('accuracy')

ax4.set_ylabel('sharpness')
plt.show()

