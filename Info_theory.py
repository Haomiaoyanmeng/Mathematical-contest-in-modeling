import numpy as np
import random

N = 1000000
row_sto = np.zeros(N)
row_cal = np.array([[-1, 0]])

for i in range(N):
    j = 1
    while random.random() < 0.5:
        j = j + 1
    row_sto[i] = j

for i in range(N):
    for j in range(len(row_cal)):
        if row_cal[j][0] == row_sto[i]:
            row_cal[j][1] = row_cal[j][1] + 1
            break
        elif j == len(row_cal) - 1:
            row_cal = np.row_stack((row_cal, [row_sto[i], 1]))

row_cal = row_cal[1:, :]
print(row_sto)
print(row_cal)
row_cal[:, 1] = row_cal[:, 1] / N
H = np.sum(row_cal[:, 1] * np.log2(row_cal[:, 1]))
print(-H)
