import matplotlib.pyplot as plt
import numpy as np

with open('./output_csv/loss_acc_seni_test.csv', 'r') as f:
    datas = f.readlines()

    batch_size = np.zeros(len(datas))
    loss_train = np.zeros(len(datas))
    loss = np.zeros(len(datas))
    acc_train = np.zeros(len(datas))
    acc = np.zeros(len(datas))
    sensi = np.zeros(len(datas))

for i, data in enumerate(datas):
    batch_size[i], loss_train[i], loss[i], acc_train[i], acc[i], sensi[i] = list(map(float,data.rstrip('\n').split(',')))


fig = plt.figure(1)
ax = fig.add_subplot(211)

ax2 = ax.twinx()

ax.semilogx(batch_size, loss_train, 'orange', label='train loss')
ax.semilogx(batch_size, loss, 'orange', linestyle='--', label='test loss')
ax.set_ylim((-0.5,1))

ax2.semilogx(batch_size, sensi, 'g-', label='sensitivity')
fig.legend(loc=2, bbox_to_anchor=(0,1), bbox_transform=ax.transAxes)

ax.set_xlabel('batch_size (log)')
ax.set_ylabel('cross entropy')

ax2.set_ylabel('sensitivity')
plt.draw()
###

ax3 = fig.add_subplot(212)
ax4 = ax3.twinx()

ax3.semilogx(batch_size, acc_train, 'orange', label='train acc')
ax3.semilogx(batch_size, acc, 'orange', linestyle='--', label='test acc')
ax3.set_ylim((0.95,1.01))

ax4.semilogx(batch_size, sensi, 'g-', label='sensitivity')

h1, l1 = ax3.get_legend_handles_labels()
h2, l2 = ax4.get_legend_handles_labels()

ax3.legend(handles=h1+h2, labels=l1+l2, loc=3, bbox_to_anchor=(0,0), bbox_transform=ax3.transAxes)

ax3.set_xlabel('batch_size (log)')
ax3.set_ylabel('accuracy')

ax4.set_ylabel('sensitivity')

plt.show()
