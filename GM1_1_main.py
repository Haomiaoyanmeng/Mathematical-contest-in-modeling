import matplotlib.pyplot as plt
import numpy as np

# 原始数据
A = [174, 179, 183, 189, 207, 234, 220.5, 256, 270, 285]
# 级比检验
A = np.array(A)
n = len(A)
min0 = np.exp(-2 / (n + 1))
max0 = np.exp(2 / (n + 1))
sigma = np.zeros(n - 1)
sumA = np.zeros(n)  # 这里的累加序列有n个
Z = np.zeros(n - 1)  # 而这里的均值只有n-1个

# 注意range后面到end-1, 而不是end, 这个很重要
for i in range(1, n):
    sigma[i - 1] = A[i - 1] / A[i]

# 这里最好学习一下怎么删除向量中的某个元素
ii = 0  # 记录级比检验不对的个数
for i in range(2, n+1):
    if not min0 < sigma[i - 2] < max0:
        ii = ii + 1
        print('第%d个级比不在标准区间内\n', i)
print('不符合标准数所占百分比为%f', ii / (n - 1))
# 对向量做累加  累加完成之后就是n-1个  不知道有无现成函数
sumA[0] = A[0]
for i in range(1, n):
    sumA[i] = sumA[i - 1] + A[i]

# 生成紧邻均值序列
for i in range(0, n - 1):
    Z[i] = (sumA[i] + sumA[i + 1]) / 2

B = np.column_stack((-Z, np.ones(n - 1)))
Y = A
Y = np.delete(Y, 0)

# print(type(B), type(Y))  # 这里B是list的形式不能运算
B = np.array(B)  # 这里B为何有三个维度???
# print(np.size(-Z, 0))
# print(np.size(B, 0))
# print(B)
# C = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
C = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
# print(np.size(C, 0))
a = C[0]
b = C[1]
# print(a)
# print(C)  # 原来C一直忘记加乘Y了
#

# # 预测写法一
# # 先预测
# F = [A[0]]
# for i in range(1, n + 9):
#     F.append((A[0] - b / a) / np.exp(a * i) + b / a)  # append 是往数组后面接着加
# # print(F)
# # 后累减还原  F和G都是
# G = [A[0]]
# for i in range(1, n + 9):
#     G.append(F[i] - F[i - 1])
# # print(G)

# 预测写法二
F = np.zeros(n+10)
print(np.size(F))
F[0] = A[0]
for i in range(1, n + 10):
    F[i] = ((A[0] - b / a) / np.exp(a * i) + b / a)  # append 是往数组后面接着加
print(F)
# 后累减还原  F和G都是
G = np.zeros(n+10)
G[0] = A[0]
for i in range(1, n + 10):
    G[i] = (F[i] - F[i - 1])
print(G)

plt.plot(A)
plt.plot(G)
#
plt.show()
