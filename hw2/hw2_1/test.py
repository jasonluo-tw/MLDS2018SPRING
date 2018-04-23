import os

aa = os.popen('ls ./MLDS_hw2_1_data/training_data/feat/').read().split()

for i in aa[:5]:
    print(i.rstrip('.npy'))
