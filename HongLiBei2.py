import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import cm

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def Judgement1(N_loc1, row, col, N):  # 返回点
    # N^2个数字换到一行上面
    N_sto = np.zeros([N ** 2, 2])
    flag = 0
    for row1 in range(N):
        for col1 in range(N):
            N_sto[flag][0] = cos_sim(N_loc1[row][col], N_loc1[row1][col1])  # 验证之后发现这一行可以这样进行输入
            N_sto[flag][1] = np.sqrt((N_loc1[row][col][0] - N_loc1[row1][col1][0]) ** 2 +
                                     (N_loc1[row][col][1] - N_loc1[row1][col1][1]) ** 2 +
                                     (N_loc1[row][col][2] - N_loc1[row1][col1][2]) ** 2)
            flag = flag + 1
    """
     上面是计算完了 之后要判断是否正相关
     先排序, 我把BubbleSort改成了两列同时进行的排序
    """
    # N_sto2 = BubbleSort(N_sto)
    # flag1 = 1  # 判断是否正相关
    # for i in range(N_sto2.shape[0] - 1):
    #     if N_sto2[i][1] > N_sto2[i + 1][1]:
    #         flag1 = 1
    #     else:
    #         flag1 = 0
    #         break
    return N_sto


def Judgement2(N_loc1, row, col, N):  # 返皮尔逊系数
    # N^2个数字换到一行上面
    N_sto = np.zeros([N ** 2, 2])
    flag = 0
    for row1 in range(N):
        for col1 in range(N):
            N_sto[flag][0] = cos_sim(N_loc1[row][col], N_loc1[row1][col1])  # 验证之后发现这一行可以这样进行输入
            N_sto[flag][1] = np.sqrt((N_loc1[row][col][0] - N_loc1[row1][col1][0]) ** 2 +
                                     (N_loc1[row][col][1] - N_loc1[row1][col1][1]) ** 2 +
                                     (N_loc1[row][col][2] - N_loc1[row1][col1][2]) ** 2)
            flag = flag + 1
    """
     上面是计算完了 之后要判断是否正相关
     先排序, 我把BubbleSort改成了两列同时进行的排序
    """
    pr = pearsonr(N_sto[:, 0], N_sto[:, 1])
    # N_sto2 = BubbleSort(N_sto)
    # flag1 = 1  # 判断是否正相关
    # for i in range(N_sto2.shape[0] - 1):
    #     if N_sto2[i][1] > N_sto2[i + 1][1]:
    #         flag1 = 1
    #     else:
    #         flag1 = 0
    #         break
    return pr[0]


# 下面是一个冒泡排序的算法
def BubbleSort(arr):
    for i in range(1, arr.shape[0]):
        for j in range(0, arr.shape[0] - i):
            if arr[j][1] > arr[j + 1][1]:  # 按着第二列距离进行排序
                arr[j][0], arr[j + 1][0] = arr[j + 1][0], arr[j][0]
                arr[j][1], arr[j + 1][1] = arr[j + 1][1], arr[j][1]
    return arr


# 计算两个向量的余弦相似度
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def plot1(N):
    # 先遍历每一个图和其他子图的相关性和距离  存储在N1中  其中距离不好计算
    N_loc = np.zeros([N, N, 3])
    # 构建画三维图用的小坐标
    for row in range(N):
        for col in range(N):
            N_loc[row][col][0] = -(N - 1) / 2 + row
            N_loc[row][col][1] = (N - 1) / 2 - col
            N_loc[row][col][2] = N ** 2
    return Judgement1(N_loc, 0, 0, N)


def plot2(N):
    # 先遍历每一个图和其他子图的相关性和距离  存储在N1中  其中距离不好计算
    N_loc = np.zeros([N, N, 3])
    # 构建画三维图用的小坐标
    for row in range(N):
        for col in range(N):
            N_loc[row][col][0] = -(N - 1) / 2 + row
            N_loc[row][col][1] = (N - 1) / 2 - col
            N_loc[row][col][2] = N ** 2
    return Judgement2(N_loc, 0, 0, N)


def Sns2nd(N_loc1, A, B):
    """
    说是画热图用的, 其实这是用我们的计算方法
    来进行计算评判指标的函数
    :param N_loc1:
    :param A:
    :param B:
    :return:
    """
    C = np.zeros([4, 4])
    for row in range(4):
        for col in range(4):
            # C[row][col] = np.linalg.norm(N_loc1[row][col] - N_loc1[0][0]) + np.sqrt(A[row][col]) + np.sqrt(B[row][col])
            C[row][col] = np.sqrt((N_loc1[0][0][0] - N_loc1[row][col][0]) ** 2 +
                                  (N_loc1[0][0][1] - N_loc1[row][col][1]) ** 2 +
                                  (N_loc1[0][0][2] - N_loc1[row][col][2]) ** 2 +
                                  (A[0][0] - A[row][col]) ** 2 +
                                  (B[0][0] - B[row][col]) ** 2)
    return C


def High_de_RN(A, B, n):
    N = 4
    N_sto = np.zeros([N ** 2, 2])
    N_loc2 = np.zeros([N, N, n])  # 存储"位置"向量
    for row in range(4):
        for col in range(4):
            N_loc2[row][col][0] = -(N - 1) / 2 + row
            N_loc2[row][col][1] = (N - 1) / 2 - col
            N_loc2[row][col][2] = N ** (n - 2)
            if n == 4:
                N_loc2[row][col][3] = A[row][col]  # 这里就不需要再开根号了
            elif n == 5:
                N_loc2[row][col][3] = A[row][col]
                N_loc2[row][col][4] = B[row][col]
    flag = 0
    for row1 in range(N):
        for col1 in range(N):
            N_sto[flag][0] = cos_sim(N_loc2[row1][col1], N_loc2[0][0])  # 这里就是把其他向量和第一个子图进行计算
            if n == 4:
                N_sto[flag][1] = np.sqrt((N_loc2[0][0][0] - N_loc2[row1][col1][0]) ** 2 +
                                         (N_loc2[0][0][1] - N_loc2[row1][col1][1]) ** 2 +
                                         (N_loc2[0][0][2] - N_loc2[row1][col1][2]) ** 2 + # np.sqrt(A[row1][col1])# + np.sqrt(B[row1][col1])
                                         (N_loc2[0][0][3] - N_loc2[row1][col1][3]) ** 2
                                         )
            elif n == 5:
                N_sto[flag][1] = np.sqrt((N_loc2[0][0][0] - N_loc2[row1][col1][0]) ** 2 +
                                         (N_loc2[0][0][1] - N_loc2[row1][col1][1]) ** 2 +
                                         (N_loc2[0][0][2] - N_loc2[row1][col1][2]) ** 2 +  # np.sqrt(A[row1][col1])# + np.sqrt(B[row1][col1])
                                         (N_loc2[0][0][3] - N_loc2[row1][col1][3]) ** 2 +
                                         (N_loc2[0][0][4] - N_loc2[row1][col1][4]) ** 2
                                         )

            flag = flag + 1

    return N_sto


N = 4
N_test = np.zeros([N, N])
N1 = np.zeros([N, 2])
# 先遍历每一个图和其他子图的相关性和距离  存储在N1中  其中距离不好计算
N_loc = np.zeros([N, N, 3])
# 构建画三维图用的小坐标
xx = np.zeros(N)
yy = np.zeros(N)
for row in range(N):
    for col in range(N):
        N_loc[row][col][0] = -(N - 1) / 2 + row
        xx[row] = -(N - 1) / 2 + row
        N_loc[row][col][1] = (N - 1) / 2 - col
        yy[row] = -xx[row]
        N_loc[row][col][2] = N ** 2
print(N_loc)

'''
之后就是计算相关程度
1.先找每个子图对应计算和其他子图的向量相关程度和距离存储在N1
2.冒泡排序排序出每个值
3.判断是否满足
'''

for row in range(N):
    for col in range(N):
        '''
        为了不让for太多再写一个函数
        Judge
        '''
        N_test[row][col] = Judgement2(N_loc, row, col, N)
print(N_test)

"""
N_test 已经求出, 之后就是绘制三维图形
"""
X, Y = np.meshgrid(xx, yy)
# print(X, xx, Y, yy)
N_test = np.abs(N_test)

"""
下面是热图绘制(皮尔逊相关系数)
"""
# N = 6
# N_test2 = np.zeros([N, N])
# N1 = np.zeros([N, 2])
# # 先遍历每一个图和其他子图的相关性和距离  存储在N1中  其中距离不好计算
# N_loc = np.zeros([N, N, 3])
# # 构建画三维图用的小坐标
# xx = np.zeros(N)
# yy = np.zeros(N)
#
# for row in range(N):
#     for col in range(N):
#         N_loc[row][col][0] = -(N - 1) / 2 + row
#         xx[row] = -(N - 1) / 2 + row
#         N_loc[row][col][1] = (N - 1) / 2 - col
#         yy[row] = -xx[row]
#         N_loc[row][col][2] = N**2
#
# for row in range(N):
#     for col in range(N):
#         '''
#         为了不让for太多再写一个函数
#         Judge
#         '''
#         N_test2[row][col] = Judgement2(N_loc, row, col, N)
# N_test2 = -N_test2
# f, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
# sns.heatmap(N_test, ax=ax1, annot=True, fmt='.4f', cmap='YlGnBu', vmin=0.964, vmax=0.970)
# sns.heatmap(N_test2, ax=ax2, annot=True, fmt='.4f', cmap='YlGnBu', vmin=0.964, vmax=0.970)
# # plt.title('$N = 5$时每个子图和其他子图的皮尔逊相关系数')
# plt.savefig("D:\弘理杯数模竞赛\ 1.jpg", dpi=1000)  # 这样子能输出一个像素很高的图片
# plt.show()


"""
之后是画出第二问的热图
这是用的自己的方法即:
距离和其他指标开根号来进行
为了避免直接再写一个函数, 因为只画二维和三维的所以就函数可以偷懒写一下
"""
T_in = np.array([[14, 5, 9, 8],
               [12, 3, 10, 1],
               [11, 2, 4, 6],
               [7, 16, 13, 15]])
E_in = np.array([[8, 5, 9, 3],
               [6, 1, 15, 12],
               [13, 11, 2, 7],
               [10, 4, 14, 16]])
T_in = np.sqrt(T_in)
E_in = np.sqrt(E_in)
# 先画二维的
C1 = Sns2nd(N_loc, T_in, np.zeros([4, 4]))
sns.heatmap(C1, annot=True, fmt='.2f', annot_kws={'size': 15}, cmap=sns.cubehelix_palette(as_cmap=True))
plt.title('具有时间关系的($t_i$)的二维输入数据热图')
plt.savefig("D:\弘理杯数模竞赛\具有时间关系的($t_i$)的二维输入数据热图(修正).svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()
C2 = Sns2nd(N_loc, T_in, E_in)
sns.heatmap(C2, annot=True, annot_kws={'size': 15}, fmt='.2f', cmap=sns.cubehelix_palette(as_cmap=True))
plt.title('具有时间关系的($t_i$)和能量关系($e_i$)三维输入数据热图')
plt.savefig("D:\弘理杯数模竞赛\具有时间关系的($t_i$)和能量关系($e_i$)三维输入数据热图(修正).svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()

# 画完热图之后画N的对比的图
N_sto_4D = High_de_RN(T_in, np.zeros([4, 4]), 4)
plt.plot(N_sto_4D[:, 1], N_sto_4D[:, 0], '.')
plt.grid()
plt.xlabel('多模态信息综合因数 $Q$', size=13)
plt.ylabel('余弦值 $cos$' + chr(952) + '$_{i,j}$', size=13)
plt.title('具有时间关系($t_i$)的二维输入数据的 $cos$' + chr(952) + '$_{i,j}$ 和 $Q$的关系', size=13)
plt.savefig("D:\弘理杯数模竞赛\具有时间关系($t_i$)的二维输入数据的 $cos$' + chr(952) + '$_{i,j}$ 和 $Q$的关系.svg", dpi=1000)
plt.show()
N_sto_4D = High_de_RN(T_in, E_in, 5)
plt.plot(N_sto_4D[:, 1], N_sto_4D[:, 0], '.')
plt.grid()
plt.xlabel('多模态信息综合因数 $Q$', size=13)
plt.ylabel('余弦值 $cos$' + chr(952) + '$_{i,j}$', size=13)
plt.title('具有时间关系($t_i$)和能量关系($e_i$)三维输入数据的 $cos$' + chr(952) + '$_{i,j}$ 和 $Q$的关系', size=13)
plt.savefig("D:\弘理杯数模竞赛\具有时间关系($t_i$)和能量关系($e_i$)三维输入数据的 $cos$' + chr(952) + '$_{i,j}$ 和 $Q$的关系.svg", dpi=1000)
plt.show()

"""
证明成立
画一张三位热图
"""
# A = [1.5, 1.5, 16, 0, np.sqrt(3)]
# B = [-1.5, 2.5, 16, 0, np.sqrt(5)]
# ti = np.linspace(1, 4, 100)
# tj = np.linspace(1, 4, 100)
# A = np.array(A)
# B = np.array(B)
# Z = np.zeros([len(ti), len(ti)])
# tt1, tt2 = np.meshgrid(ti, tj)
# for i in range(len(ti)):
#     for j in range(len(tj)):
#         A[3] = ti[i]
#         B[3] = tj[j]
#         Z[i][j] = cos_sim(A, B)
# tt1, tt2 = np.meshgrid(ti, tj)
# ax = plt.axes(projection='3d')
# ax.plot_surface(tt1, tt2, Z, rstride=4, cstride=4, cmap=cm.jet)
# ax.plot3D([1, 1], [1, 1], [Z[0][Z.shape[0] - 1], 1.0001], 'k->') #绘制带o折线
# ax.plot3D([1, 1], [1, 4.3], [Z[0][Z.shape[0] - 1], Z[0][Z.shape[0] - 1]], 'k->') #绘制带o折线
# ax.plot3D([1, 4.3], [1, 1], [Z[0][Z.shape[0] - 1], Z[0][Z.shape[0] - 1]], 'k^-') #绘制带o折线
# plt.show()
# 下面的语句没有用到~~~~~~~
# ax.set_xlabel('X')
# ax.set_xlim(-40, 40)
# ax.set_ylabel('Y')
# ax.set_ylim(-40, 40)
# ax.set_zlabel('Z')
# ax.set_zlim(-100, 100)

"""
下面是画三维的柱状图, 很丑!!!!
"""
# X, Y = X.ravel(), Y.ravel()  # 矩阵扁平化
# N_test = N_test.ravel()
# bottom = np.zeros_like(X)  # 设置柱状图的底端位值
# fig1 = plt.figure()
# ax1 = fig1.gca(projection='3d')  # 三维坐标轴
# ax1.set_zlim(0, 1)
# width = height = 1
# ax1.bar3d(X, Y, bottom, width, height, N_test, shade=True)


"""
下面是画三维的点图, 不好看出来谁高谁低
"""
# fig1 = plt.figure()
# ax1 = plt.axes(projection='3d')  # 三维坐标轴
# ax1.set_zlim(0.94, 0.97)
# ax1.scatter(X, Y, N_test)


"""
下面是画不同的N的典型图
为了避免重复计算， 把最开始的计算过程也写成了函数
"""
# N_plot1 = [3, 5, 10]  # 想要画的N的值可以随便输入
# fig2 = plt.figure()
# for i in range(len(N_plot1)):
#     N_plot1_return = plot1(N_plot1[i])
#     plt.plot(N_plot1_return[:, 1], N_plot1_return[:, 0], '.', label='$N$ = '+str(N_plot1[i]))
# plt.xlabel('距离')
# plt.ylabel('相关性')
# plt.title('距离和相关性的关系')

"""
下面是能画不同的N时， pr变化的图像
"""
# # N_plot2 = np.arange(np.log10(4), 4.2, 0.2)
# # N_plot2_sto = np.zeros(len(N_plot2))
# # for i in range(len(N_plot2)):
# #     N_plot2_sto[i] = plot2(int(10 ** N_plot2[i]))
# # 计算出的值
# N_plot2 = [0.60205999, 0.80205999, 1.00205999, 1.20205999, 1.40205999, 1.60205999,
#            1.80205999, 2.00205999, 2.20205999, 2.40205999, 2.60205999, 2.80205999,
#            3.00205999, 3.20205999, 3.40205999, 3.60205999, 3.80205999, 4.00205999]
# N_plot2_sto = [-0.96055034, -0.96639916, -0.97014634, -0.97173177, -0.9729007,  -0.97353214,
#                -0.97391067, -0.97415288, -0.9743056,  -0.97440106, -0.97446144, -0.97449928,
#                -0.97452329, -0.97453842, -0.97454796, -0.97455398, -0.97455777, -0.97456016]
# N_plot2_sto = np.array(N_plot2_sto)  # np.array是可以的,  np.mat.就不行呜呜呜呜呜
# N_plot2_sto = abs(N_plot2_sto)
# plt.plot(N_plot2, N_plot2_sto)
# print(N_plot2, '\n', N_plot2_sto)  # 输出具体的结果值
# plt.xlabel('$log_{10} N$')
# plt.ylabel('位置特征信息系数 $R_N$')
# plt.title('$N$ 与位置特征信息系数 $R_N$ 的关系')


# 这里检验（0，0，N^2）点和其他点的关系
# N_sto2 = Judgement(N_loc, 0, 0)
# x = N_sto2[:, 1]
# y = N_sto2[:, 0]
# plt.plot(x, y, '.')
# plt.xlabel('距离')
# plt.ylabel('相关性')
# plt.title('距离和相关性的关系')

# pccs = pearsonr(x, y)
# print(pccs[0])

# plt.plot()
plt.legend(loc="lower right")
plt.grid(True)  # 显示网格线

plt.show()
