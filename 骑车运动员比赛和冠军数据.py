import numpy as np
import matplotlib.pyplot as plt
import xlrd
# todo 对excel的操作
# plt.style.use('ggplot')
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def time_reverse(x1):
    y1 = np.zeros(max(x1.shape))
    for i in range(len(y1)):
        y1[i] = x1[i][0] * 3600 + x1[i][1] * 60 + x1[i][2]
    return y1

# todo 打开excle
xl = xlrd.open_workbook('D:\数学建模美赛\运算数据\比赛和冠军的数据.xlsx')
# print(xl.read())
# todo 通过索引获取工作表
table = xl.sheets()[0]
print(table)
# 获取一共多少行
rows = table.nrows
print(rows)
cols = table.ncols
print(cols)

# todo 获取第一列的整列的内容
col = table.col_values(0, 1)
print(col)

x_y = np.zeros([rows - 1, cols])
# # x_y[:, 0] = table.col_values(0)
# # x_y[:, 1] = table.col_values(1)
#
# col = table.col_values(2)
# x_y[:, 0] = np.array(col[1:])
# col = table.col_values(3)
# x_y[:, 1] = np.array(col[1:])

# 将excel里面的数据全部填满之后就可以任意进行读取了
for i in range(cols):
    col = table.col_values(i)
    x_y[:, i] = np.array(col[1:])
print(x_y)

plt.figure(figsize=(6, 5))

x1 = time_reverse(x_y[:, 2:5])
y1 = x_y[:, 1]
plt.plot(x1, y1, 'x', label='$Man$ $Race$ $Data$', c='#39b7cb')  # 淡蓝色

x_aran = np.arange(1, max(x1) + 1000, 1)
y_aran = 0.01169 * x_aran + 2.339
plt.plot(x_aran, y_aran, label='$Man$ $Fitting$ $Curve$', linewidth=2, c='#2e317c')  # 深蓝色

x2 = time_reverse(x_y[:, 7:10])
y2 = x_y[:, 6]
plt.plot(x2, y2, 'x', label='$Woman$ $Race$ $Data$', c='#ed9db2')  # 淡红色

x_aran2 = np.arange(1, max(x2) + 1000, 1)
y_aran2 = 0.009458 * x_aran2 + 1.777
plt.plot(x_aran2, y_aran2, label='$Woman$ $Fitting$ $Curve$', linewidth=2, c='#cf3553')  # 深红色

plt.xlabel('$Finish$ $time$ $of$ $champion(s)$', size=13)
plt.ylabel('$Race$ $Length(km)$', size=13)
plt.title('$Power$ $Curve$', size=13)
plt.grid(True)
plt.legend(fontsize=11)
plt.savefig("D:\数学建模美赛\图片\男女运动员的竞赛时间和距离的拟合曲线.svg", dpi=1000)  # 这样子能输出一个像素很高的图片


plt.show()
plt.figure(figsize=(6, 5))
plt.plot(np.log10(x_aran), x_aran, 'k')
# plt.title('$Logarithmic$ $transformation$ $curve$ ')
plt.savefig("D:\数学建模美赛\图片\对数转换曲线.svg", dpi=1000)  # 这样子能输出一个像素很高的图片
plt.show()
