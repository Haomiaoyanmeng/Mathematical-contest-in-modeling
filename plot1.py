import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-np.pi, np.py, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

plt.plot(X, C)
plt.plot(X, S)

plt.show()
