import numpy as np
import matplotlib.pyplot as plt

"""
需要一个二分法计算 z=w*e^w 的反函数的值
"""


def f0(x):
    return x * np.exp(x)


def ProductLog(x):
    a = 1  # 幂次 小
    b = 1  # 幂次 大
    a0 = 0  # 搜寻的小的值的初始值
    b0 = 0  # 大的

    # 搜寻小值
    while f0(a0) > x:
        a0 = -1 * (1 - 0.5 ** a)
        a = a + 1
        # print('a0=', a0, 'a=', a, 'f0=', f0(a0), 'x=', x)
    # 大的
    while f0(b0) < x:
        b0 = 2 ** b
        b = b + 1

    while not (-1e-5 < f0((a0 + b0) / 2) - x < 1e-5):
        if f0((a0 + b0) / 2) > x:
            b0 = (b0 + a0) / 2
        elif f0((a0 + b0) / 2) < x:
            a0 = (b0 + a0) / 2
        else:
            return (a0 + b0) / 2

    return (a0 + b0) / 2


def inverseFunction(x, P0):
    return np.log(1 - P0) / (1 + ProductLog((-1 - x) / np.e))  # 这个里面不知道为啥少了一个负号???


P = 0.5
alpha = 5
p = np.arange(0.01, 0.99, 0.001)
f = np.zeros(len(p))

for i in range(len(p)):
    x0 = -alpha * P / (p[i] * (1 - P))
    f[i] = inverseFunction(x0, P)

plt.plot(p, f)

plt.show()




