import numpy as np
import math

W = 30
lp = 5

Poss_pdf = np.zeros(W + 1)
Poss_pdf[0] = np.exp(-lp)
for i in range(1, W + 1):
    Poss_pdf[i] = Poss_pdf[i-1] + (lp ** i * np.exp(-lp)/math.factorial(i))

Gamma_L = []  # 定义存储的变量下限
Gamma_H = []  # 定义存储的上限
# 用A, B, C三个量来进行上下限的卡值
A = 0  # 中间值
B = 0
C = 0
for i in range(1, W + 1):
    A = 1  # 中间值
    B = 1
    C = 1
    for j in range(1, i + 1):
        A = A * Poss_pdf[int(W / j)]
        B = B * (Poss_pdf[int(W / j)] - Poss_pdf[int(W / (i * j))])
        C = C * Poss_pdf[int(W / (i * j))]
    Gamma_L.append((A - B + C) / 2)
    Gamma_H.append(A - 1 * B)

Sum_L = np.sum(Gamma_L)
Sum_H = np.sum(Gamma_H)

print('Sum_L=', Sum_L)
print(Gamma_L)
print('Sum_H=', Sum_H)
print(Gamma_H)

