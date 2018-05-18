import numpy as np
import sys
import random

a = np.array([1])
b = [1]

print("np array memory: ", sys.getsizeof(a))
print("list memory: ", sys.getsizeof(b))

lista = [1, 2, 3, 4, 5]
random.shuffle(lista)
print(lista)
