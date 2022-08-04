
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from matplotlib import cm

plt.figure()

ax = plt.axes(projection='3d')

ax.set_xlim(0, 2) # X轴，横向向右方向

ax.set_ylim(0, 2) # Y轴,左向与X,Z轴互为垂直

ax.set_zlim(0, 2) # 竖向为Z轴

# z = np.linspace(0, 4*np.pi, 500)
#
# x = 10*np.sin(z)
#
# y = 10*np.cos(z)
#
# ax.plot3D(x, y, z, 'black') # 绘制黑色空间曲线
#
# z1 = np.linspace(0, 4*np.pi, 500)
#
# x1 = 5*np.sin(z1)
#
# y1 = 5*np.cos(z1)
#
# ax.plot3D(x1,y1,z1,'g--') #绘制绿色空间虚曲线

# ax = plt.gca()  # get current axis 获得坐标轴对象
# ax.spines['right'].set_color('none')  # 将右边 边沿线颜色设置为空 其实就相当于抹掉这条边
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# # 设置中心的为（0，0）的坐标轴
# ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
# ax.spines['left'].set_position(('data', 0))


#
# ax.plot3D([0, 5], [0, 10], [0, 5], 'b-') #绘制带o折线
u = np.linspace(0, 2, 100)
x, y = np.meshgrid(u, u)
z = - (x - 1) * (y - 1) + 1
ax.plot_surface(x, y, z, rstride=4, cstride=4, cmap=cm.jet)
ax.plot3D([0, 0], [0, 0], [0, 2.2], 'k->') #绘制带o折线
ax.plot3D([0, 0], [0, 2.2], [0, 0], 'k->') #绘制带o折线
ax.plot3D([0, 2.2], [0, 0], [0, 0], 'k^-') #绘制带o折线


# ax.plot3D([5], [10], [5], 'b>') #绘制带o折线
# ax.set_axis_off()
ax.grid()
ax.grid(visible=True)
# ax.quiver([1], [1], [1], [2], [2], [2], length=0.1, normalize=True)
plt.show()


