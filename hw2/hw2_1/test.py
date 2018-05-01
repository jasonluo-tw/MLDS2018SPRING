import os
import numpy as np
aa = os.popen('ls ./MLDS_hw2_1_data/training_data/feat/').read().split()

for i in aa[:5]:
    print(i.rstrip('.npy'))

aa = [0,1,2,3,4,5]
bb = np.arange(0,2)

cc = {'a':5, 'b':6, 'c':7}

print([cc[i] for i in ['a', 'b', 'c']])

print('###')
for i in range(10):
    if i >= 5 and i <= 7:
        continue
    print(i)

aa = ''
for i in range(10):
    aa = aa + ''.join('c')
    print(aa)
