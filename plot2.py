import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 1000)  # 这个是设置数组0~10 一共1000个点
print(len(x))

plt.figure(figsize=(8, 4))
# 这个是画的图像大小

y1 = np.sin(x)
y2 = np.cos(x**2)
plt.plot(x, y1, label="$sin(x)$", color="red", linewidth=2)
plt.plot(x, y2, "b--", label="$cos(x^2)$")

plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("Pylot ")

plt.ylim(-2, 2)  # 这个是显示的横纵坐标范围
plt.legend()
plt.grid()

plt.show()

# x = np.zeros([10, 2])
# print(x)
