import numpy as np
import math
import cmath as cs
import matplotlib.pyplot as plt
import seaborn as sns
import random

# # 两种经常用的定义方法一定要掌握
x = np.arange(-5, 5, 2)
print(x)
# x = np.linspace(-5, 5, 101)
# print(x)

# 这个有助于理解函数变量啥的
from numpy.core._multiarray_umath import ndarray

# def change(a):
#     a = 10
#     print(a)
#
#
# a = 1
# print(a)
# change(a)
# print(a)
#
# arfa = 0.1
# str0 = str(int(100 * arfa))
# print(str0)
#
# for i in range(2, 5):
#     print(i)

# 如何去除矩阵里边的元素
# a = [0, 0, 0]
a = np.zeros(3)
b = list(a)
b.remove(b[0])
print(b)

# b = np.arange(2, 5, 1)  # 这个输出居然不包含5
# print(b)
# print(b[0])
# print(b[1])
#
for i in range(-1, 5):
    print(i)
#
# # 矩阵运算
# # 这波是彻底懂了怎么设置向量啥的
# a = np.array([[1, 2, 3, 4]])
# b = np.array([[1], [2], [3], [4]])
# c = a * b.T
# print(c)
# print(c.T)
# c = a.dot(b)
# print(c)
# a = [[1, 2, 3, 4]]
# b = [[1], [2], [3], [4]]
# a = np.array(a)
#
# c = a * b.T

# b = [1, 2, 3, 4]
# a = np.array([[1, 1, 1, 1]])
# a[0, 1] = b[2]
# print(a)
# a = np.zeros(5)
# a[2] = b[2]
# print(a)

W = 30
lp = 10

Poss_pdf = []
for i in range(0, W):
    Poss_pdf.append((lp ** i * np.exp(-lp)) / math.factorial(i))

# print('i=', i)
# print(lp ^ i)
j = 1
for i in range(0, j):
    print('yyy', i)

print(cs.sqrt(1))
if cs.sqrt(1).imag == 0:
    print('hhh')

a = [1, 2, 2, 2, 3, 4, 5]
a = np.delete(a, len(a) - 1)
print(a)

# x = Symbol('x')
# y = Symbol('y')
# C = solve([3*x + 4*y - np.e, 8*x - y - 14], [x, y])
# C = list(C)
# print(x)

np.random.seed(0)
x = np.random.randn(4, 4)
f, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)  # ncols   nrows  分别指的是行中和列中有多少个图
sns.heatmap(x, annot=True, ax=ax1, label='1')
sns.heatmap(x, annot=True, fmt='.1f', ax=ax2, label='2')
plt.legend()
# plt.show()
plt.figure(dpi=120)

print(x, x)
# x = [1, 2, 3]
# y = [2, 4, 6]
# pr = pearsonr(x, y)
# print(pr)

x = [[1, 1, 0], [2, 2, 2]]
print(np.linalg.norm(x[0]))

plt.quiver([0, 0], [1, 1])
# plt.show()

ti = np.linspace(1, 4, 100)
print(ti)

x = [[1, 2], [1, 2, 3]]
print(x)

# with open("D:/数学建模美赛/K_means算法数据.txt", "r") as f:  # 打开文件
#     data = f.read()  # 读取文件
#     print(data)
#
# data = np.mat(data)
# print(data)
# x = []
# for i in data:
#     x.append(i)
# print(x)
data = np.genfromtxt("D:/数学建模美赛/K_means算法数据.txt", dtype=[float, float])  # 将文件中数据加载到data数组里
data = np.array(data)
print(data)
dataset = np.loadtxt("D:/数学建模美赛/K_means算法数据.txt")
print(dataset)

a = np.array([-2 + 1j, 2])
b = np.abs(a)
print(b)
print(max(b))

a = [1 for i in range(3)]
print(a)
a = [[1 for i in range(3)] for j in range(3)]
print(a)
a = [0] * (3)
print(a)

a = np.array(range(0, 100, 1))

print(a[1:3])

print(np.log(2.713))

a = b = 0
print(a, b)
a = np.array([1, 5, 4, 7, 8, 6, 9, 5, 4, 0])
a.sort()
print(a, a[:0])
print(random.random() * 300 + 300)

a = np.array([[1, 2], [3, 4], [5, 6]])
print(a.shape[0], a[[0, 2], :])
print(np.random.rand(5, 1))

s = np.array([1, -1])
print((s < 0) * 1.1 + (s > 0) * s)
s = -1
print((s < 0) * 1.1 + (s > 0) * 2.2)

print(a * a / a)

b = np.array([1, 2])
c = np.array(np.vstack((a, b, a)))
print(c)
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([1, 2])
c = np.append(a, b)
print(c)
print(np.cos(np.pi / 2))

for i in range(1, 0):
    print(a[0][0])

a = np.array([1, 2])
b = np.array([3, 4])
print(a * b)

a = [[]]
print((a == []) * 2)
b = np.zeros([3, 2])
# b[1] = a

print('sss', 0 * np.log(np.e))
print(np.log2(2))

a = np.array([[4, 4], [5, 5]])

c = np.row_stack((a, [8,9]))
d = np.column_stack((a, [8,9]))
print('aaa', c, d)

print(3000 / np.log2(1.01))
