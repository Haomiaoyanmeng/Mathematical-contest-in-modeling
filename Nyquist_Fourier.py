import numpy as np
import matplotlib.pyplot as plt
import random

def Nyquist(arfa, Ts, len1):
    # Ts = 1
    # len1 = 1000
    w = np.linspace(-2 * 2 * np.pi / Ts, 2 * 2 * np.pi / Ts, len1)
    y = np.zeros(w.size)
    t = np.linspace(-5 * Ts, 5 * Ts, len1)
    h_real = np.zeros(t.size)
    h_im = np.zeros(t.size)
    h = np.zeros(t.size)

    # 先建立频域的函数
    # 设置滚降系数
    # arfa = 0
    # for i in range(w.size):
    #     if (0.25 - arfa) * len1 < i < (0.25 + arfa) * len1:
    #         y[i] = (i - (0.25 - arfa) * len1) / (2 * arfa * len1)
    #     elif (0.25 + arfa) * len1 <= i < (0.75 - arfa) * len1:
    #         y[i] = 1
    #     elif (0.75 - arfa) * len1 <= i < (0.75 + arfa) * len1:
    #         y[i] = ((0.75 + arfa) * len1 - i) / (2 * arfa * len1)
    #     else:
    #         y[i] = 0

    # # 这个是余弦滚降
    # for i in range(w.size):
    #     if (0.25 - arfa) * len1 < i < (0.25 + arfa) * len1:
    #         beta = 1-(i-(0.25-arfa)*len1)/(2*arfa*len1)
    #         y[i] = 0.5+0.5*np.cos(-np.pi*beta)
    #     elif (0.25 + arfa) * len1 <= i < (0.75 - arfa) * len1:
    #         y[i] = 1
    #     elif (0.75 - arfa) * len1 <= i < (0.75 + arfa) * len1:
    #         beta = (i - (0.75 - arfa)*len1)/(2*arfa*len1)
    #         y[i] = 0.5+0.5*np.cos(np.pi*beta)
    #     else:
    #         y[i] = 0

    # 再之后是纯噪声
    for i in range(w.size):
        if 0 < i < len1/2:
            y[i] = random.random()
            y[500 + i] = 1 - y[i]

    for i in range(t.size):
        for j in range(w.size):
            h_real[i] = h_real[i] + 1 / (2 * np.pi) * y[j] * np.cos(w[j] * t[i])
            h_im[i] = h_im[i] + 1 / (2 * np.pi) * y[j] * np.sin(w[j] * t[i])

    for i in range(t.size):
        h[i] = np.sqrt(h_real[i] ** 2 + h_im[i] ** 2)

    str0 = "$alpha$=0." + str(int(1000 * arfa))  # 这样子好麻烦, 有无更简洁的方法
    plt.plot(w, y)
    # plt.plot(t, h)


Nyquist(0, 1, 1000)
# Nyquist(0.125, 1, 1000)
# Nyquist(0.25, 1, 1000)
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("$h(t)$")

# print(y)
# print(h)
# plt.legend()
plt.show()
