import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nf

# x = np.linspace(-10000, 10000, 20001)
# y = np.zeros(20001)
# for i in range(x.size):
#     if -50 < i < 50:
#         y[i] = 1


# plt.figure('FFT', facecolor='lightgray')
# plt.subplot(121)
# plt.title('Time Domain', fontsize=16)
# plt.grid(linestyle=':')
# plt.plot(x, y, label=r'$y$')
# # 针对方波y做fft
# comp_arr = nf.fft(y)
# y2 = nf.ifft(comp_arr).real
# plt.plot(x, y2, color='orangered', linewidth=5, alpha=0.5, label=r'$y$')
# # 绘制频域图形
# plt.subplot(122)
# freqs = nf.fftfreq(y.size, x[1] - x[0])
# pows = np.abs(comp_arr)  # 复数的模
# plt.title('Frequency Domain', fontsize=16)
# plt.grid(linestyle=':')
# plt.plot(freqs[freqs > 0], pows[freqs > 0], color='orangered', label='frequency')

# plt.legend()
# plt.savefig('fft.png')
plt.show()

print(x.size)
print(y)

