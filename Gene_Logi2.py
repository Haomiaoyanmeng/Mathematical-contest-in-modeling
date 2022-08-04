# -*-coding:utf-8 -*-
# 目标求解2*sin(x)+cos(x)最大值
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import xlrd


# todo 对excel的操作


# 初始化生成chromosome_length大小的population_size个个体的二进制基因型种群
def species_origin(population_size, chromosome_length):
    kind = 0
    kind_stage = np.array([[349.4, 438361.1, 1384], [252.76, 70891.2, 963.2], [256.5, 116141.2, 1074.2]])
    min_gen = kind_stage[kind][0] + 30
    max_gen = kind_stage[kind][0] + 110
    population = [[]]
    # 二维列表，包含染色体和基因
    for i in range(population_size):
        temporary = []
        # 染色体暂存器
        for j in range(chromosome_length):
            temporary.append(random.random() * min_gen + (max_gen - min_gen))
            # 随机生成一个从min到max的一个浮点数
        population.append(temporary)
        # 将染色体添加到种群中
    return population[1:]  # 之前里面是1:  想不太明白
    # 将种群返回，种群是个二维数组，个体和染色体两维


# # 目标函数相当于环境 对染色体进行筛选，这里是用自己的公式来计算  即先计算时间, 之后再进行限制, 限制之后再给他进行惩罚
# def function(population, chromosome_length, max_value):
#     temporary = []
#     function1 = []
#     temporary = translation(population, chromosome_length)
#     # 暂存种群中的所有的染色体(十进制)
#     for i in range(len(temporary)):
#         x = temporary[i] * max_value / (math.pow(2, chromosome_length) - 1)
#         # 一个基因代表一个决策变量，其算法是先转化成十进制，然后再除以2的基因个数次方减1(固定值)。
#         function1.append(2 * math.sin(x) + math.cos(x))
#         # 这里将2*sin(x)+cos(x)作为目标函数，也是适应度函数
#     return function1


def crossover(population, pc):
    # pc是概率阈值，加一个交叉的阈值吧
    pop_len = len(population)

    for i in range(pop_len - 1):
        if random.random() < pc:  # 以pc为概率发生交叉
            cpoint = random.randint(0, len(population[0]))
            # 在种群个数内随机生成单点交叉点
            temporary1 = []
            temporary2 = []

            temporary1.extend(population[i][0:cpoint])
            temporary1.extend(population[i + 1][cpoint:len(population[i])])
            # 将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
            # 然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面

            temporary2.extend(population[i + 1][0:cpoint])
            temporary2.extend(population[i][cpoint:len(population[i])])
            # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
            # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
            population[i] = temporary1
            population[i + 1] = temporary2
            # 第i个染色体和第i+1个染色体基因重组/交叉完成
    return population


# step4：突变
def mutation(population, pm):
    # pm是概率阈值  思想是生成一个种群个数个向量
    px = len(population)
    # 求出种群中所有种群/个体的个数
    py = len(population[0])
    # 染色体/个体中基因的个数

    population_sto = [[]]  # 存储变异基因

    for i in range(px):
        temporary = []
        if random.random() < pm:
            # 如果小于阈值就变异
            mpoint = random.randint(0, py - 1)
            # 生成0到py-1的随机数
            temporary = population[i]
            temporary[mpoint] = temporary[mpoint] + 40 * random.uniform(-0.5, 1)
            population_sto.append(temporary)
    return population_sto[1:]


# 下面是一个冒泡排序的算法
def BubbleSort(a, b):  # 这里就规定按照a进行排序, 并且是小的在前面
    for i in range(1, a.shape[0]):
        for j in range(0, a.shape[0] - i):
            if a[j] > a[j + 1]:  # 按着第二列距离进行排序
                a[j], a[j + 1] = a[j + 1], a[j]
                b[j], b[j + 1] = b[j + 1], b[j]
    return a, b


# 3.选择种群中个体适应度最大的个体
def selection(population, fitness1):
    fitness_sort, population_sort = BubbleSort(fitness1, population)
    return population_sort[:population_size], fitness_sort[:population_size]


def W_v(W_in, s, sita):  # 输入功率, 和阶段stage,
    n = 0.7
    m = 70
    g = 9.8
    Cr = 0.025
    mw = 4.6
    Cd = 0.92
    A = 0.8
    rou = 1.29
    Vw = 0.5 * np.cos(sita)
    P = W_in
    # 后面的s是坡度
    a = (s > 0) * 0 + (s < 0) * (-0.05 * s)

    A1 = 0.5 * Cd * A * rou / n
    B = m * g / n * (Cr + s / 100 + a / g * (1 + mw / m)) + Vw ** 2
    C = -P

    V = (- B + np.sqrt(B ** 2 - 4 * A1 * C)) / (2 * A1)
    return V


def fitness_cal(population, length_road, s, sita):
    time_road = np.zeros(len(population[0]))
    fitness = np.zeros(population.shape[0])
    cons_store = np.zeros(population.shape[0])  # 来存储是否满足限制条件即是否消耗完艳阳储备
    # 进行列的循环把每一个个体来进行评判
    for i in range(population.shape[0]):
        V = W_v(population[i], s, sita)
        # 求出速度之后求出总时间
        for j in range(len(population[0])):
            time_road[j] = length_road[j] / V[j]
        fitness[i] = sum(time_road)

        t_max = fitness[i]  # 求得总时间, 做下面的仿真
        dt = 1
        t = np.arange(dt, t_max, dt)
        P = np.zeros(len(t))
        t_sto = 0  # 存储时间, 来进行矩形波的划分
        t_stage = 0  #
        for k in range(len(t)):  # 先生成方波
            t_sto = t_sto + dt
            if t_sto < sum(time_road[:(t_stage + 1)]) + 2 * dt:  # 加了一点点防止出各种bug
                t_stage = t_stage
            else:
                t_stage = t_stage + 1
            P[k] = population[i][t_stage]  # 给第k个时间来进行赋值

        kind = 0
        kind_stage = np.array([[348.43, 438361.1, 1384], [249.86, 70891.2, 963.2], [256.5, 116141.2, 1074.2]])
        Wa = np.zeros(len(t))
        Wa[0] = kind_stage[kind][1]
        bate = [3.5, 3, 3.5]
        tao_W = 300000  #
        for i1 in range(1, len(t)):
            Wa1 = bate[kind] * (kind_stage[kind][2] - kind_stage[kind][0]) * (kind_stage[kind][0] - P[i1]) * np.exp((kind_stage[kind][1] - Wa[i1 - 1]) / tao_W) \
                  / (bate[kind] * kind_stage[kind][2] - P[i1])  # 这里invaild的bug是明明是一维的数组, 结果有一个二维的索引
            Wa[i1] = Wa[i1 - 1] + dt * Wa1
            if Wa[i1] < 0 or Wa[i1] > kind_stage[kind][1] + 5000:
                cons_store[i] = 1
                break

    # 之后就是最后计算fitness
    for i in range(len(fitness)):
        fitness[i] = fitness[i] + 1000000 * cons_store[i]

    return fitness, V, len(t)


def distance_P(length_road, P):  # 后面输入一个个体
    V = W_v(P, s, sita)
    # 求出速度之后求出总时间
    time_road = np.zeros(len(P))
    for j in range(len(P)):
        time_road[j] = length_road[j] / V[j]

    t_max = sum(time_road)  # 求得总时间, 做下面的仿真
    dt = 1
    t = np.arange(dt, t_max, dt)
    P_stot = np.zeros(len(t))
    t_sto = 0  # 存储时间, 来进行矩形波的划分
    t_stage = 0  #
    for k in range(len(t)):  # 先生成方波
        t_sto = t_sto + dt
        if t_sto < sum(time_road[:(t_stage + 1)]) + 2 * dt:  # 加了一点点防止出各种bug
            t_stage = t_stage
        else:
            t_stage = t_stage + 1
        P_stot[k] = P[t_stage]  # 给第k个时间来进行赋值

    kind = 0
    kind_stage = np.array([[349.4, 438361.1, 1384], [252.76, 70891.2, 963.2], [256.5, 116141.2, 1074.2]])
    Wa = np.zeros(len(t))
    Wa[0] = kind_stage[kind][1]
    bate = [3.5, 3, 3.5]  # 这里记得要上下来一起修改
    tao_W = 300000  #
    for i1 in range(1, len(t)):
        Wa1 = bate[kind] * (kind_stage[kind][2] - kind_stage[kind][0]) * (kind_stage[kind][0] - P_stot[i1]) * np.exp((kind_stage[kind][1] - Wa[i1 - 1]) / tao_W) \
              / (bate[kind] * kind_stage[kind][2] - P_stot[i1])  # 这里invaild的bug是明明是一维的数组, 结果有一个二维的索引
        Wa[i1] = Wa[i1 - 1] + dt * Wa1  # 这里注意是P_stot 这里是用的后一个,

    x_max = sum(length_road)  # 求得总时间, 做下面的仿真
    dx = 1
    x = np.arange(dx, x_max, dx)
    P_sto = np.zeros(len(x))
    x_sto = 0  # 存储时间, 来进行矩形波的划分
    x_stage = 0  #
    for k in range(len(x)):  # 先生成方波
        x_sto = x_sto + dx
        if x_sto < sum(length_road[:(x_stage + 1)]) + 2 * dx:  # 加了一点点防止出各种bug
            x_stage = x_stage
        else:
            x_stage = x_stage + 1
        P_sto[k] = P[x_stage]  # 给第k个时间来进行赋值

    return x, P_sto, t, Wa


# todo 打开excle
xl = xlrd.open_workbook('D:\数学建模美赛\运算数据\map3.xlsx')
# print(xl.read())
# todo 通过索引获取工作表
table = xl.sheets()[0]
print(table)
# 获取一共多少行
rows = table.nrows
print(rows)

# todo 获取第一列的整列的内容
col = table.col_values(0, 1)
print(col)

x_y = np.zeros([rows - 1, 2])
# x_y[:, 0] = table.col_values(0)
# x_y[:, 1] = table.col_values(1)
col = table.col_values(1)
length_road = np.array(col[1:]) * 1000
col = table.col_values(2)
s = np.array(col[1:])
col = table.col_values(3)
sita = np.array(col[1:])
print(length_road)

# 要读取length_road, s, sita

population_size = 50
generations = 25
max_value = 10
# 基因中允许出现的最大值
chromosome_length = 8  # 这个是路段数量, 每一段就是对应着一个功率
pc = 1  # 以我们的算法这个值最好是1
pm = 0.2


results = []
fitness1 = []

population = species_origin(population_size, chromosome_length)
population = np.array(population)
print(population.shape[0])
# 生成一个初始的种群

for i in range(generations):  # 注意这里是迭代500次

    # function1 = function(population, chromosome_length, max_value)
    # 首先是交叉互换产生子代
    population_cro = crossover(population, pc)  # 交配\
    population_cro = np.array(population_cro)
    print(population_cro.shape[0])
    population_mut = mutation(population, pm)
    population_mut = np.array(population_mut)
    print(population_mut.shape[0])
    # 三个矩阵合并成为一个, 解决这个神奇的bug
    population_all = np.zeros(
        [population.shape[0] + population_cro.shape[0] + population_mut.shape[0], population.shape[1]])
    population_all[:population.shape[0]] = population
    population_all[population.shape[0]: population.shape[0] + population_cro.shape[0]] = population_cro
    if population != [[]]:
        population_all[population.shape[0] + population_cro.shape[0]:] = population_mut

    fitness1, V, test = fitness_cal(population_all, length_road, s, sita)
    print(fitness1, V, test, 'hhh')

    population, fitness1 = selection(population_all, fitness1)  # 选择
    print(fitness1, fitness1.shape, 'kkk')
    best_individual, best_fitness = population[0], fitness1[0]
    results.append(fitness1[0])
    # # 将最好的个体和最好的适应度保存，并将最好的个体转成十进制
    print('generation=', i)

print(results)
print(V, '\n', best_individual)

# 画出length和P的关系
x, P, t, Wa = distance_P(length_road, best_individual)

# 第一个图迭代的代数和时间关系  长
plt.figure(figsize=(7, 5))
plt.plot(results,  color="black", linewidth=2)  # , marker='o', markerfacecolor='None'
plt.xlabel('$Generation$', size=13)
plt.ylabel('$Time(s)$', size=13)
plt.title('$Genetic$ $Algorithm$', size=13)
plt.grid(True)
plt.savefig("D:\数学建模美赛\图片\迭代的时间和代数.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(x, P, linewidth=2, c='#2e317c')
plt.xlabel('$Distance(m)$', size=13)
plt.ylabel('$Power(J)$', size=13)
plt.title('$Power$ $Distribution$', size=13)
plt.grid(True)
plt.savefig("D:\数学建模美赛\图片\距离和功率.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(t, Wa, linewidth=2, c='#cf3553')
plt.xlabel('$Time(s)$', size=13)
plt.ylabel('$Wa(J)$', size=13)
# plt.title('$Time$ $Wa$', size=13)
plt.grid(True)
plt.savefig("D:\数学建模美赛\图片\时间和Wa.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()

# plt.plot(best_individual)
# plt.show()




