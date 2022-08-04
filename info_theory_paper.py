import numpy as np
import matplotlib.pyplot as plt

tao = [3, 6, 8, 10]
p = np.arange(0.01, 0.99, 0.0001)

for i in range(len(tao)):
    plt.plot(p, ((1 - p) ** tao[i]) * (1 - p) / p - (1 - p) / p)

plt.show()

# p = 0.5
# tao = np.arange(3, 20, 0.01)
#
# for i in range(tao):
#     i = 100

for i in range(2):
    plt.plot(p, (1 - p) ** tao[i])
    plt.plot(p, 1 - tao[i] * p)

plt.ylim([0, 1])
plt.xlim([0, 0.15])
plt.show()

P = 0.5
plt.plot(p, (1 - np.log(1 - P) / p) * (1 - P) ** (1 / p) - 1)

plt.show()

p = -0.01
print(p, (1 - np.log(1 - P) / p), (1 - P) ** (1 / p) - 1)
