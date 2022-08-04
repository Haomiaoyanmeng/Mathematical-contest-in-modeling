import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

# 通过read_csv来读取我们的目的数据集
# 我去因为创建的是xlsx文件用的read_csv读不出来, read_excel 就出来了
adv_data = pd.read_excel("D:/数学建模美赛/算法多元回归的数据.xlsx")   # , encoding='gb2312', error_bad_lines=False

# 清洗不需要的数据
# new_adv_data = adv_data.ix[:,1:]
new_adv_data = adv_data
# 得到我们所需要的数据集且查看其前几列以及数据形状
print('head:', adv_data.head(), '\nShape:', adv_data.shape)

# 数据描述
print(new_adv_data.describe())
# 缺失值检验
print(new_adv_data[new_adv_data.isnull() == True].count())

new_adv_data.boxplot()
plt.savefig("boxplot.jpg")
plt.show()
# 相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
# 相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
print(new_adv_data.corr())

# 通过加入一个参数kind='reg'，seaborn可以添加一条最佳拟合直线和95%的置信带。
# zcy: 想改一下x_vars发现不会改23333
sns.pairplot(new_adv_data, x_vars=['TV', 'radio', 'newspapre'], y_vars='sales', height=3, aspect=1, kind='reg')
plt.savefig("pairplot.jpg")
plt.show()
# 后面的.iloc  原来是.ix  该之后才能用, 意思就是取前三个
X_train, X_test, Y_train, Y_test = train_test_split(new_adv_data.iloc[:, :3], new_adv_data.sales, train_size=.80)

print("原始数据特征:", new_adv_data.iloc[:, :3].shape,
      ",训练数据特征:", X_train.shape,
      ",测试数据特征:", X_test.shape)

print("原始数据标签:", new_adv_data.sales.shape,
      ",训练数据标签:", Y_train.shape,
      ",测试数据标签:", Y_test.shape)

model = LinearRegression()
model.fit(X_train, Y_train)
a = model.intercept_  # 截距
b = model.coef_  # 回归系数
print("最佳拟合线:截距", a, ",回归系数：", b)
# y=2.668+0.0448∗TV+0.187∗Radio-0.00242∗Newspaper


# R方检测
# 决定系数r平方
# 对于评估模型的精确度
# y误差平方和 = Σ(y实际值 - y预测值)^2
# y的总波动 = Σ(y实际值 - y平均值)^2
# 有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
# 有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
# 对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
# 2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
score = model.score(X_test, Y_test)
print(score)

# 对线性回归进行预测
Y_pred = model.predict(X_test)
print(Y_pred)

plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
# 显示图像
# plt.savefig("predict.jpg")
plt.show()

plt.figure()
plt.plot(range(len(Y_pred)), Y_pred, 'b', label="predict")
plt.plot(range(len(Y_pred)), Y_test, 'r', label="test")
plt.legend(loc="upper right")  # 显示图中的标签
plt.xlabel("the number of sales")
plt.ylabel('value of sales')
# plt.savefig("ROC.jpg")
plt.show()





