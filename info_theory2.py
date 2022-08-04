import numpy as np
import matplotlib.pyplot as plt

x1 = [-np.pi/4, np.pi/4, np.pi]
y1 = [0, 0, 0]
area1 = np.array([1 / 3, 1 / 3, 1 / 3])
area1 = area1 * 15  # 没有这个就画的特别小...
area1 = np.pi * area1 ** 2  # 转化成面积

x2 = [0, 0]
y2 = [2 ** 0.5 / 2, -1]
area2 = np.array([2 / 3, 1 / 3])
area2 = area2 * 15  # 没有这个就画的特别小...
area2 = np.pi * area2 ** 2  # 转化成面积

x3 = np.arange(-np.pi / 2, np.pi * 4 / 3, 0.01)
y3 = np.cos(x3)

# definitions for the axes
left, width = 0.1, 0.65  # 位置和大小
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
plt.grid()
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

ax.plot(x3, y3)
ax.scatter(x1, np.cos(x1), s=30)
ax_histx.scatter(x1, y1, s=area1, c='b', alpha=0.5, marker='o')
ax_histy.scatter(x2, y2, s=area2, c=['#2e317c', '#8abce1'], alpha=0.5, marker='o')
# 去除一些坐标轴
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)

plt.show()


x1 = [np.pi/4, np.pi/2, np.pi]
y1 = [0, 0, 0]
area1 = np.array([1 / 3, 1 / 3, 1 / 3])
area1 = area1 * 15  # 没有这个就画的特别小...
area1 = np.pi * area1 ** 2  # 转化成面积

x2 = [0, 0]
y2 = [2 ** 0.5 / 2, -1]
area2 = np.array([2 / 3, 1 / 3])
area2 = area2 * 15  # 没有这个就画的特别小...
area2 = np.pi * area2 ** 2  # 转化成面积

x3 = np.arange(-np.pi / 2, np.pi * 4 / 3, 0.01)
y3 = np.cos(x3)

# definitions for the axes
left, width = 0.1, 0.65  # 位置和大小
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
plt.grid()
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

ax.plot(x3, y3)
ax.scatter(x1, np.cos(x1), s=30)
ax_histx.scatter(x1, y1, s=area1, c='b', alpha=0.5, marker='o')
ax_histy.scatter([0] * 3, np.cos(x1), s=area1, c='b', alpha=0.5, marker='o')
# 去除一些坐标轴
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)

plt.show()


x1 = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
y1 = np.zeros(len(x1))
area1 = np.array([5 / 15, 4 / 15, 3 / 15, 2 / 15, 1 / 15])  # 其实无所谓
area1 = area1 * 20  # 没有这个就画的特别小...
area1 = np.pi * area1 ** 2  # 转化成面积

x2 = np.zeros(len(x1))
y2 = pow(2, x1)
area2 = np.array([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5])
area2 = area2 * 20  # 没有这个就画的特别小...
area2 = np.pi * area2 ** 2  # 转化成面积

x3 = np.arange(-3, 2.5, 0.01)
y3 = pow(2, x3)

# definitions for the axes
left, width = 0.1, 0.65  # 位置和大小
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
plt.grid()
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

ax.plot(x3, y3)
ax.scatter(x1, 2 ** x1, s=30)
ax_histx.scatter(x1, y1, s=area1, c='b', alpha=0.5, marker='o')
ax_histy.scatter(x2, y2, s=area1, alpha=0.5, marker='o')
# 去除一些坐标轴
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)

plt.show()
