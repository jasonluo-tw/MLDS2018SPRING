import matplotlib.pyplot as plt
import numpy as np

xx = np.linspace(0,1,50000)

y1 = np.sin(20 * np.pi * xx) * np.tan(2 * np.pi * xx) / (2 * np.pi * xx + 0.5)

y2 = np.sign(np.sin(7 * np.pi * xx))

y3 = 2 * np.sin(20 * np.pi * xx) * np.cos(5 * np.pi * xx)

plt.subplot(131)

plt.plot(xx,y1)

plt.subplot(132)

plt.plot(xx,y2)

plt.subplot(133)

plt.plot(xx,y3)

plt.show()
