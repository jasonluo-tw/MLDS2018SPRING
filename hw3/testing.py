import numpy as np
import sys
import random

a = np.array([1])
b = [1]

print("np array memory: ", sys.getsizeof(a))
print("list memory: ", sys.getsizeof(b))

lista = [1, 2, 3, 4, 5]

random.Random(5).shuffle(lista)
#print(lista)

lista = [5, 4, 3, 2, 1]
random.Random(5).shuffle(lista)
#print(lista)

from random import randint
print(randint(1, 4))

tags00 = [1, 1] + [0] * 20

tags0 = []
tags0 = [tags00 for i in range(20)]

#tags01 = np.array(tags0)
print(len(tags0))
