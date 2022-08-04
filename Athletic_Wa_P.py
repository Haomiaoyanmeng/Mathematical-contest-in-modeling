import numpy as np
import matplotlib.pyplot as plt
import xlrd
# todo 对excel的操作
# plt.style.use('ggplot')
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# pre:         distance  T_c     P_c    P_max   CP     W1
pra = np.array([[44.2,  3581.5,	348.43,	1384,   256.4, 438361.1],
                [22.1,	2157.8,	249.86,	963.2,  219.5, 70891.2],
                [43.1,  3495.2, 349.4,  1384,   256.4, 438361.1],
                [19.3,  1844.2, 252.76, 963.2,  219.5, 70891.2]])

# 画图的参数
t_max = 4100  # 4 * 60 * 60  # 单位是秒
dt = 0.01
t = np.arange(dt, t_max, dt)
# print(t)
t_log = np.log10(t)

T_pre = 10  # 前段加速的时间
arfa = 0.017

name = ['$Man$', '$Woman$', '$Man$', '$Woman$']
color = ['#8abce1', '#ec8aa4', '#005CAF', '#00AA90']

# # 下面这种是产生的几种方式
# P = np.zeros([len(t), 4])
# for j in range(4):
#     for i in range(len(t)):
#         # if t_x[i] < T_pre:  # 衰减方式1
#         #     P[i][j] = pra[j][3] * np.exp(- (t_x[i] / T_pre) * np.log(pra[j][3] / pra[j][2]))
#         # else:
#         #     P[i][j] = pra[j][2] * np.exp(- arfa * (t_x[i] - T_pre))
#
#         # if t[i] < T_pre:  # 衰减方式2
#         #     P[i][j] = pra[j][3] - ((pra[j][3] - arfa * pra[j][5] / T_pre - pra[j][2]) / T_pre) * t[i]
#         # else:
#         #     P[i][j] = arfa * pra[j][5] / t[i] + pra[j][2]
#
#         if (int(t[i] / 100) % 2) == 0:
#             P[i][j] = pra[j][2] + 100
#         else:
#             P[i][j] = pra[j][2] - 100
# plt.plot(t, P[:, 0])
# plt.show()

# 下面画三种周期的图像主要是不同周期和幅度
P = np.zeros([len(t), 4])
Vpp = [100, 100, 50]
T = np.array([160, 80, 160]) * 0.8  # 这里是半个周期
num_T = [10, 20, 10]
t_flag = 0
t_store = 0
# k = 6  # 一种要画几个(半周期)
k1 = 0
for i in range(len(t)):  # 之前不知道是怎么了在这一喊语句之后再加一个赋值语句  但是结构就老是为0 不太明白
    # P[i][0] = pra[0][2]
    if t_store < T[t_flag]:  # 是否在一个  半周期内
        if (k1 % 2) == 0:
            P[i][0] = pra[0][2] + Vpp[t_flag] / 2
        else:
            P[i][0] = pra[0][2] - Vpp[t_flag] / 2
        t_store = t_store + dt
    elif k1 < num_T[t_flag] - 1:  # 不在的话就判断一下是不是要换下一种
        k1 = k1 + 1
        t_store = 0
        if (k1 % 2) == 0:
            P[i][0] = pra[0][2] + Vpp[t_flag] / 2
        else:
            P[i][0] = pra[0][2] - Vpp[t_flag] / 2
    elif t_flag < 2:
        t_flag = t_flag + 1
        k1 = 0
        t_store = 0
        if (k1 % 2) == 0:
            P[i][0] = pra[0][2] + Vpp[t_flag] / 2
        else:
            P[i][0] = pra[0][2] - Vpp[t_flag] / 2
    elif t_flag == 2:  # 跳出循环
        t_store = 0
        for j in range(int(sum(T * np.array(num_T)) / dt), len(t)):
            P[j][0] = pra[0][2] - Vpp[t_flag] / 2
        break

Pc_store = np.zeros(len(t))  # 和下面的语句是画两条直线
Wa_store = np.zeros(len(t))
for i in range(len(t)):
    Pc_store[i] = pra[0][2]
    Wa_store[i] = pra[0][5]

color = ['#8abce1', '#ec8aa4', '#2e317c', '#cf3553']  # 浅蓝, 浅红, 深蓝, 深红

fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(111)
ax1.plot(t, P[:, 0], c=color[0], linewidth=2, label='$P$')
ax1.set_ylabel('$Power(W)$', fontdict={'weight': 'normal', 'size': 15}, c='#3a8fb7')
ax1.set_xlabel('$Time(s)$', fontsize=15)
ax1.tick_params(axis='y', labelcolor='#3a8fb7')
ax1.set_title("$Square$ $Function$", fontdict={'weight': 'normal', 'size': 15})
plt.plot(t, Pc_store, '-.', linewidth=2, c=color[2], label='$P_c$')
plt.legend(fontsize=16, bbox_to_anchor=[0.80, 0.203])

Wa = np.zeros(len(t))
Wa[0] = pra[0][5]
bate = [0.7, 2]
tao_W = 20000
for i in range(1, len(t)):
    Wa1 = bate[0] * (pra[0][3] - pra[0][2]) * (pra[0][2] - P[i][0]) * np.exp((pra[0][5] - Wa[i - 1]) / tao_W)\
          / (bate[0] * pra[0][3] - P[i][0])
    Wa[i] = Wa[i - 1] + dt * Wa1

ax2 = ax1.twinx()  # this is the important function
ax2.plot(t, Wa, c=color[3], linewidth=2, label='$W_a$')
ax2.set_ylabel('$W_a(J)$', fontdict={'weight': 'normal', 'size': 15}, c='r')
ax2.set_xlabel('Same')
ax2.tick_params(axis='y', labelcolor='r')
# ax2.set_ylim(ymin=390000)
ax2.set_xlim(0, t_max)
plt.grid(True)
# # 参数rotation设置坐标旋转角度，参数fontdict设置字体和字体大小
# ax1.set_xticklabels(df['yearmonth'],rotation=90,fontdict={'weight': 'normal', 'size': 15})
plt.plot(t, Wa_store, '-.', linewidth=2, c='k', label='$E_a$')
print(len(Pc_store), len(t), '\n', Pc_store)
plt.legend(fontsize=15, bbox_to_anchor=[0.9101, 0.2])
plt.savefig("D:\数学建模美赛\图片\矩形波与Wa的升降曲线.svg", dpi=1000)  # 这样子能输出一个像素很高的图片

plt.show()

