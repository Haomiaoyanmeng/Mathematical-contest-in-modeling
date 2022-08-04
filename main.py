import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
# X = np.linspace(-np.pi, np.pi, 256, endpoint=True)# -π to+π的256个值
C, S = np.cos(X), np.sin(X)

plt.plot(X, C)
plt.plot(X, S)

plt.show()

max(3, 2)
print(max(3, 2))

a = 1 / 8
a = float(1)
print(a)


def genetic(a, b):
    if a > b:
        return a
    else:
        return b


print(genetic(3, 2))

a = 3.1111111111111111
b = 3.1111111111111112
print((a < b) + 1)
# float??精度可以到小数点之后15位16位就不行了
a = 1e-17
b = 2e-17
print((a < b) + 1)
# print(pi)


