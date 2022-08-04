import numpy as np
import math

W = 30
lp = 1

Poss_pdf = np.zeros(W + 1)
Poss_pdf[0] = np.exp(-lp)
for i in range(1, W + 1):
    Poss_pdf[i] = Poss_pdf[i-1] + (lp ** i * np.exp(-lp)/math.factorial(i))

Gamma_n = []
b = 1  # 中间值
for i in range(0, W):
    b = 1
    for j in range(0, i + 1):
        b = b * Poss_pdf[int(W / (math.pow(math.factorial(i+1), 1/(i + 1)) * (j + 1)))]
    Gamma_n.append(b)

Sum = np.sum(Gamma_n)
print(Sum)
print(Gamma_n)

