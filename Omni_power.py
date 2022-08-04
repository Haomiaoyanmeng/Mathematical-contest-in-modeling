import numpy as np
import matplotlib.pyplot as plt
import xlrd
# todo 对excel的操作
# plt.style.use('ggplot')
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 画图的参数
t_max = 4 * 60 * 60  # 4 * 60 * 60  # 单位是秒
t_log = np.log10(t_max)  # 取个对数
# print(10 ** t_log, t_log)
dt = 0.001
t = np.arange(dt, t_log, dt)
# print(t)
t_x = 10 ** t

# # Omni数据
# x_y = [[1500,	5],
#        [690,	60],
#        [500,	120],
#        [400,	2400],
#        [380,	3600]]
# x_y = np.array(x_y)

# # 三参数数据
# x_y = [[446,	100],
#         [385,	172],
#         [324,	434],
#         [290,	857],
#         [280,	1361]]
# x_y = np.array(x_y)
# print(x_y[:, 1])


# todo 打开excle
xl = xlrd.open_workbook('D:\数学建模美赛\运算数据\运动员2.xlsx')
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
col = table.col_values(2)
x_y[:, 0] = np.array(col[1:])
col = table.col_values(3)
x_y[:, 1] = np.array(col[1:])
print(x_y)


# # Omni模型中的参数
# W1 = 71830
# P_max = 602.2
# CP = 317.8
# CP_TTF = 40 * 60  # 后面两个参数没有
# A = 0.6
#
# P = np.zeros(len(t))
# for i in range(len(t)):
#     if t_x[i] <= CP_TTF:
#         P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max - CP) / W1))) + CP
#     elif t_x[i] > CP_TTF:
#         P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max - CP) / W1))) + CP - A * np.log(t_x[i] / CP_TTF)

#
# # 3-pra模型
# CP = 260.3
# W1 = 27409.9
# P_max = 1003.8
#
# P = np.zeros(len(t))
# for i in range(len(t)):
#     P[i] = 1 / ((t_x[i] / W1) + 1 / (P_max - CP)) + CP

# AP模型
k = 0.04
P_max = 826.4  # 770
P_3min = 463.4  # 445

CP_TTF = 40 * 60
W1 = 71830
P_max_O = 510  # 原数据602.2
CP = 317.8
A = 0.6

t_spli = 100
P = np.zeros(len(t))
for i in range(len(t)):
    if t_x[i] < t_spli:
        P[i] = P_3min + (P_max - P_3min) * np.exp(- k * t_x[i])
    # elif t_spli < t_x[i] <= CP_TTF:
    #     P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max_O - CP) / W1))) + CP
    # elif t_x[i] > CP_TTF:
    #     P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max_O - CP) / W1))) + CP - A * np.log(t_x[i] / CP_TTF)
    else:
        P[i] = 1 / ((t_x[i] / W1) + 1 / (P_max_O - CP)) + CP

plt.figure(figsize=(9, 5))
plt.plot(t, P, c='#2e317c', linewidth=2)
# plt.plot(np.log10(x_y[:, 1]), x_y[:, 0], 'ro')  # 把红点去掉
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Power(W)$', size=13)
plt.title('$Power$ $Curve$', size=13)
plt.grid(True)
plt.savefig("D:\数学建模美赛\图片\一张单独的运动曲线.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()

# # 不同人的数据
# boxing = np.array([715.14, 116.49, 671.92])
# basketball = np.array([1001.60, 153.52, 669.15])
# long_run = np.array([742.95, 107.22, 517.78])
# name = ['men', 'women', 'boxer', 'basketball player', 'soccer']
# color = ['#FDB251', '#41ae3c', '#005CAF', '#00AA90', '#592C63']
#
# # # for i in color:
# # #     plt.plot(exec(i), '.')  # 想玩一个花的就是用字符串表示那个, 结果没做出来hhh之后细细研究
# # plt.plot(boxing)
# # plt.plot(basketball)
# # plt.plot(long_run)
#
#
# # 不同人的参数
# k_woman = 0.9
# P_max_whole = np.array([P_max, P_max * k_woman, 0, 0, 0])
# P_max_O_whole = np.array([P_max_O, P_max_O * k_woman - 3, 5, -5, -4])
# P_3min_whole = np.array([P_3min, P_3min * k_woman, 0, 0, 0])
# CP_TTF_whole = np.array([40, 30, 25, 23, 21])
# CP_TTF_whole = CP_TTF_whole * 60
# CP_whole = np.array([317.8, 317.8 * k_woman, 317.8 * 0.7, 317.8 * 0.8, 317.8 * 0.65])
#
# Athletic = [boxing, basketball, long_run]
# print(Athletic)
# for i in range(3):
#     P_max_whole[i + 2] = Athletic[i][0] * 0.6 + Athletic[i][2] * 0.1
#     P_max_O_whole[i + 2] = P_max_O_whole[i + 2] + Athletic[i][0] * 0.3 + Athletic[i][2] * 0.05
#     P_3min_whole[i + 2] = Athletic[i][0] * 0.25 + Athletic[i][2] * 0.1
#     CP_whole[i + 2] = Athletic[i][0] * 0.2 + Athletic[i][2] * 0.1
# print(P_max_whole, '\n', P_max_O_whole, '\n', P_3min_whole, '\n', CP_whole, '\n', CP_TTF_whole)

P_max_whole = np.array([1384, 963.2, 1074.2])
P_max_O_whole = np.array([636.1, 520.9, 474.4])
P_3min_whole = np.array([594.5, 418.7, 426.4])
CP_TTF_whole = np.array([60, 60, 40])
CP_TTF_whole = CP_TTF_whole * 60
CP_whole = np.array([256.4, 219.5, 247.6])
W1_whole = np.array([438361.1, 70891.2, 116141])
name = ['$Man$', '$Woman$', '$Climber$']
color = ['#2e317c', '#cf3553', '#9fa39a']  # 蓝, 红, 灰

plt.figure(figsize=(9, 5))
for j in range(3):
    P = np.zeros(len(t))
    for i in range(len(t)):
        if t_x[i] <= t_spli:
            P[i] = P_3min_whole[j] + (P_max_whole[j] - P_3min_whole[j]) * np.exp(- k * t_x[i])
        # elif t_spli < t_x[i] <= CP_TTF_whole[j]:
        #     P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max_O_whole[j] - CP_whole[j]) / W1))) + CP_whole[j]
        # elif t_x[i] > CP_TTF_whole[j]:
        #     P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max_O_whole[j] - CP_whole[j]) / W1))) + CP_whole[j] - A * np.log(t_x[i] / CP_TTF_whole[j])
        else:
            P[i] = 1 / ((t_x[i] / W1_whole[j]) + 1 / (P_max_O_whole[j] - CP_whole[j])) + CP_whole[j]
    plt.plot(t, P, label=name[j], c=color[j], linewidth=2)
# plt.plot(np.log10(x_y[:, 1]), x_y[:, 0], 'ro')  # 把红点去掉
# np.log10(x_y[:, 1])

plt.legend()
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Power(W)$', size=13)
plt.title('$Different$ $types$ $of$ $riders\'$ $Power$ $Curve$', size=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
# plt.savefig("D:\数学建模美赛\图片\不同运动员的运动曲线.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.savefig("D:\数学建模美赛\图片\不同运动员的运动曲线.png")  # 这样子能输出一个像素很高的图片

plt.show()


# 下面偷懒了是要画竖着的图

plt.figure(figsize=(5, 7))
for j in range(3):
    P = np.zeros(len(t))
    for i in range(len(t)):
        if t_x[i] <= t_spli:
            P[i] = P_3min_whole[j] + (P_max_whole[j] - P_3min_whole[j]) * np.exp(- k * t_x[i])
        # elif t_spli < t_x[i] <= CP_TTF_whole[j]:
        #     P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max_O_whole[j] - CP_whole[j]) / W1))) + CP_whole[j]
        # elif t_x[i] > CP_TTF_whole[j]:
        #     P[i] = (W1 / t_x[i]) * (1 - np.exp(- t_x[i] * ((P_max_O_whole[j] - CP_whole[j]) / W1))) + CP_whole[j] - A * np.log(t_x[i] / CP_TTF_whole[j])
        else:
            P[i] = 1 / ((t_x[i] / W1_whole[j]) + 1 / (P_max_O_whole[j] - CP_whole[j])) + CP_whole[j]
    plt.plot(P, t, label=name[j], c=color[j], linewidth=3)
# plt.plot(np.log10(x_y[:, 1]), x_y[:, 0], 'ro')  # 把红点去掉
# np.log10(x_y[:, 1])

plt.legend()
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Power(W)$', size=13)
plt.title('$Different$ $types$ $of$ $riders\'$ $Power$ $Curve$', size=13)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=14)
plt.grid(True)
plt.savefig("D:\数学建模美赛\图片\不同运动员的运动曲线反转坐标轴.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
# ax.invert_yaxis()

plt.show()
