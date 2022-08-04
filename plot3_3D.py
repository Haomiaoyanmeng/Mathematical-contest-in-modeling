import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
x, y = np.meshgrid(x, y)
r = np.sqrt(x ** 2 + y ** 2)
z = np.sin(r)

fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(x, y, z, rstride=1, cstride=100000, cmap=cm.viridis)
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.xlim(-6, 6)
plt.ylim(-6, 6)

# ax.annotate('Beta(1,1)', xy=(0, 0), xytext=(0, 1),
#             arrowprops=dict(facecolor='black', arrowstyle='<-'))


#  ax.arrow(0, 0.25, 0.3, -0.02, head_width=0.02, head_length=0.5, shape="full",fc='red',ec='red',alpha=0.9, overhang=0.1)

plt.show()


