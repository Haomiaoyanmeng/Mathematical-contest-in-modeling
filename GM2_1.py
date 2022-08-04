import matplotlib.pyplot as plt
import numpy as np
import cmath as cs
from sympy import *

# 原始数据
A = [174, 179, 183, 189, 207, 234, 220.5, 256, 270, 285]
# 级比检验
A = np.array(A)
n = len(A)
min0 = np.exp(-2 / (n + 1))
max0 = np.exp(2 / (n + 1))
sigma = np.zeros(n - 1)
sumA = np.zeros(n)  # 这里的累加序列有n个
a_A = np.zeros(n - 1)
Z = np.zeros(n - 1)  # 而这里的均值只有n-1个

# # GM(2,1)不知怎么进行级比检验

# 生成和序列
sumA[0] = A[0]
for i in range(1, n):
    sumA[i] = sumA[i - 1] + A[i]

# 生成差序列
for i in range(0, n - 1):
    a_A[i] = A[i + 1] - A[i]

# 生成紧邻均值序列
for i in range(0, n - 1):
    Z[i] = (sumA[i] + sumA[i + 1]) / 2

B = np.column_stack((-Z, np.ones(n - 1)))
B = np.column_stack((-np.delete(A, 0), B))
Y = a_A
# Y = np.delete(Y, 0)

B = np.array(B)
C = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
a1 = C[0]
a2 = C[1]
b = C[2]

# 之后定义一下解的形式 0 为不同实根  1为相同实根  2为有虚数根
flag = 0
solve1 = (-a1 + cs.sqrt(a1 ** 2 - 4 * a2)) / 2
solve2 = (-a1 - cs.sqrt(a1 ** 2 - 4 * a2)) / 2
if solve1.imag == 0:
    if not solve1 == solve2:
        flag = 0
    else:
        flag = 1  # 实际中不会出现
else:
    flag = 2

# 实际中参数不可能碰巧, 所以只考虑不等的情况
C0 = b / a2

# 之后求解前面的参数c1, c2
# x = Symbol('x')
# y = Symbol('y')

if flag == 0:
    x = np.array([[np.exp(solve1.real * (n - 1)), np.exp(solve2.real * (n - 1))], [np.exp(0), np.exp(0)]])
    y = np.array([sumA[n - 1] - C0, sumA[0] - C0])
    c1 = (y[1] * x[0][1] - y[0] * x[1][1]) / (x[1][0] * x[0][1] - x[0][0] * x[1][1])
    c2 = (y[1] * x[0][0] - y[0] * x[1][0]) / (x[1][1] * x[0][0] - x[0][1] * x[1][0])
    # c1 = C1[0]
    # c2 = C1[1]
    # c1, c2 = solve([x * np.exp(solve1.real * n) + y * np.exp(solve2.real * n) - sumA[n - 1],
    #                 x * np.exp(solve1.real) + y * np.exp(solve2.real) - sumA[0]], [x, y])
elif flag == 2:
    x = np.array([[np.exp(-solve1.real) * np.cos(solve1.imag), np.exp(-solve1.real) * np.sin(solve1.imag)],
                 [np.exp(-solve1.real) * np.cos(solve1.imag * n), np.exp(-solve1.real) * np.sin(solve1.imag * n)]])
    y = np.array([sumA[0] - C0, sumA[n - 1] - C0])
    c1 = (y[1] * x[0][1] - y[0] * x[1][1]) / (x[1][0] * x[0][1] - x[0][0] * x[1][1])
    c2 = (y[1] * x[0][0] - y[0] * x[1][0]) / (x[1][1] * x[0][0] - x[0][1] * x[1][0])
    # c1, c2 = solve(x, y)
    # c1, c2 = solve([np.exp(-solve1.real)*(x * np.cos(solve1.imag) + y * np.sin(solve1.imag)) - sumA[0],
    #                 np.exp(-solve1.real)*(x * np.cos(solve1.imag * n) + y * np.sin(solve1.imag * n)) - sumA[n - 1]], [x, y])

print(solve1, solve2, c1, c2, sumA[n-1], sumA[0])

add_n = 3
F = np.zeros(n+add_n)
F[0] = A[0]
if flag == 0:
    for i in range(1, n + add_n):
        F[i] = c1 * np.exp(solve1.real * i) + c2 * np.exp(solve2.real * i) + C0  # append 是往数组后面接着加
elif flag == 2:
    for i in range(1, n + add_n):
        F[i] = np.exp(solve1.real) * (c1 * np.exp(solve1.real * (i - 1)) + c2 * np.exp(solve2.real * (i - 1))) + C0  # append 是往数组后面接着加
print(C0)
print(F)
# 后累减还原  F和G都是
G = np.zeros(n+add_n)
G[0] = A[0]
for i in range(1, n + add_n):
    G[i] = (F[i] - F[i - 1])


plt.plot(A)
plt.plot(G)

plt.show()

