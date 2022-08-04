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
t_max = 1 * 60 * 60  # 4 * 60 * 60  # 单位是秒
t_log = np.log10(t_max)  # 取个对数
# print(10 ** t_log, t_log)
dt = 0.001
t = np.arange(dt, t_log, dt)
# print(t)
t_x = 10 ** t

T_pre = 10  # 前段加速的时间
arfa = 0.017

name = ['$Man$', '$Woman$', '$Man$', '$Woman$']
color = ['#8abce1', '#ec8aa4', '#2e317c', '#cf3553']  # 浅蓝, 浅红, 深蓝, 深红

P = np.zeros([len(t), 4])
for j in range(4):
    for i in range(len(t)):
        # if t_x[i] < T_pre:  # 衰减方式1
        #     P[i][j] = pra[j][3] * np.exp(- (t_x[i] / T_pre) * np.log(pra[j][3] / pra[j][2]))
        # else:
        #     P[i][j] = pra[j][2] * np.exp(- arfa * (t_x[i] - T_pre))
        if t_x[i] < T_pre:  # 衰减方式2
            P[i][j] = pra[j][3] - ((pra[j][3] - arfa * pra[j][5] / T_pre - pra[j][2]) / T_pre) * t_x[i]
        else:
            P[i][j] = arfa * pra[j][5] / t_x[i] + pra[j][2]
    # plt.plot(t_x, P[:, j])
# print(P)

# 先画对数的曲线不要图例
plot_size = [20, 5]
plt.figure(figsize=plot_size)
plt.subplot(1, 2, 1)
plt.plot(t, P[:, 0], c=color[0], label=name[0], linewidth=2)
plt.plot(t, P[:, 1], c=color[1], label=name[1], linewidth=2)
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Race$ $Length(km)$', size=13)
plt.title('$Tokyo$', size=13)
plt.grid(True)
# plt.legend(fontsize=11)

plt.subplot(1, 2, 2)
plt.plot(t, P[:, 2], c=color[2], label=name[2], linewidth=2)
plt.plot(t, P[:, 3], c=color[3], label=name[3], linewidth=2)
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Race$ $Length(km)$', size=13)
plt.title('$Belgium$', size=13)
plt.grid(True)
# plt.legend(fontsize=11)

plt.savefig("D:\数学建模美赛\图片\运动员不同比赛和性别的subplot图形对数坐标.svg", dpi=1000)  # 这样子能输出一个像素很高的图片

plt.show()


# 之后画正常坐标来作为主要图形
location_legend = [0.4, 0.9]  # 先左右, 后上下, 以右和上为正方向
plt.figure(figsize=plot_size)
plt.subplot(1, 2, 1)
plt.plot(t_x, P[:, 0], c=color[0], label=name[0], linewidth=2)
plt.plot(t_x, P[:, 1], c=color[1], label=name[1], linewidth=2)
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Race$ $Length(km)$', size=13)
plt.title('$Tokyo$', size=13)
plt.grid(True)
plt.legend(fontsize=13, bbox_to_anchor=location_legend)  # loc='upper left'

plt.subplot(1, 2, 2)
plt.plot(t_x, P[:, 2], c='#2e317c', label=name[2], linewidth=2)
plt.plot(t_x, P[:, 3], c='#cf3553', label=name[3], linewidth=2)
plt.xlabel('$log_{10} $ $of$ $Time(s)$', size=13)
plt.ylabel('$Race$ $Length(km)$', size=13)
plt.title('$Belgium$', size=13)
plt.grid(True)
plt.legend(fontsize=13, bbox_to_anchor=location_legend)

plt.savefig("D:\数学建模美赛\图片\运动员不同比赛和性别的subplot图形正常坐标.svg", dpi=1000)  # 这样子能输出一个像素很高的图片

plt.show()

