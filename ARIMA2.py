# -*- coding: utf-8 -*-
# 用 ARIMA 进行时间序列预测
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.graphics.api import qqplot

# 1.创建数据
data = [101, 118, 90, 79, 108, 123, 94, 83, 109, 125, 96, 86, 113, 131, 102, 91, 119, 131, 108, 97]
# data = [5922, 5308, 5546, 5975, 2704, 1767, 4111, 5542, 4726, 5866, 6183, 3199, 1471, 1325, 6618, 6644, 5337, 7064,
#         2912, 1456, 4705, 4579, 4990, 4331, 4481, 1813, 1258, 4383, 5451, 5169, 5362, 6259, 3743, 2268, 5397, 5821,
#         6115, 6631, 6474, 4134, 2728, 5753, 7130, 7860, 6991, 7499, 5301, 2808, 6755, 6658, 7644, 6472, 8680, 6366,
#         5252, 8223, 8181, 10548, 11823, 14640, 9873, 6613, 14415, 13204, 14982, 9690, 10693, 8276, 4519, 7865, 8137,
#         10022, 7646, 8749, 5246, 4736, 9705, 7501, 9587, 10078, 9732, 6986, 4385, 8451, 9815, 10894, 10287, 9666, 6072,
#         5418]

data = pd.Series(data)
data.index = pd.Index(sm.tsa.datetools.dates_from_range('1901', '1920'))  # 这里要改横轴坐标  玄学与长度相关
data.plot(figsize=(12, 8))
# 绘制时序的数据图
plt.show()

# 2.下面我们先对非平稳时间序列进行时间序列的差分，找出适合的差分次数d的值：
# fig = plt.figure(figsize=(12, 8))
# ax1 = fig.add_subplot(111)
# diff1 = data.diff(1)
# diff1.plot(ax=ax1)
# 这里是做了1阶差分，可以看出时间序列的均值和方差基本平稳，不过还是可以比较一下二阶差分的效果：

# 这里进行二阶差分
# fig = plt.figure(figsize=(12, 8))
# ax2 = fig.add_subplot(111)
# diff2 = data.diff(2)
# diff2.plot(ax=ax2)
# 由下图可以看出来一阶跟二阶的差分差别不是很大，所以可以把差分次数d设置为1，上面的一阶和二阶程序我们注释掉

# 这里我们使用一阶差分的时间序列
# 3.接下来我们要找到ARIMA模型中合适的p和q值：
data1 = data.diff(1)
data1.dropna(inplace=True)
# 加上这一步，不然后面画出的acf和pacf图会是一条直线

# 第一步：先检查平稳序列的自相关图和偏自相关图
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data1, lags=10, ax=ax1)  # zcy:这个里面最一开始又一个lags=40 是一共要画的点数, 不过总的点数要比实际总的数据长度要少很多, 不知具体关系
# lags 表示滞后的阶数
# 第二步：下面分别得到acf 图和pacf 图
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data1, lags=10, ax=ax2)

# 由上图可知，我们可以分别用ARMA(7,0)模型、ARMA(7,1)模型、ARMA(8,0)模型等来拟合找出最佳模型：  zcy:这里是要根据两个图来判断模型再来筛选出我们要的模型
# 第三步:找出最佳模型ARMA
arma_mod1 = sm.tsa.ARMA(data1, (1, 0)).fit()
print(arma_mod1.aic, arma_mod1.bic, arma_mod1.hqic)
arma_mod2 = sm.tsa.ARMA(data1, (1, 1)).fit()
print(arma_mod2.aic, arma_mod2.bic, arma_mod2.hqic)
arma_mod3 = sm.tsa.ARMA(data1, (2, 0)).fit()
print(arma_mod3.aic, arma_mod3.bic, arma_mod3.hqic)

# 由上面可以看出ARMA(7,0)模型最佳   zcy: 这里看着好像是要找三个数字中最小的
# 第四步：进行模型检验
# 首先对ARMA(7,0)模型所产生的残差做自相关图
resid = arma_mod1.resid
# 一定要加上这个变量赋值语句，不然会报错resid is not defined
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=10, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=10, ax=ax2)

# 接着做德宾-沃森（D-W）检验
print(sm.stats.durbin_watson(arma_mod1.resid.values))
# 得出来结果是不存在自相关性的     zcy: 这里百度说接近2就挺好的, 具体的话还要有很多范围限制

# 再观察是否符合正态分布,这里用qq图
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

# 最后用Ljung-Box检验:检验的结果就是看最后一列前十二行的检验概率（一般观察滞后1~12阶），
# 如果检验概率小于给定的显著性水平，比如0.05、0.10等就拒绝原假设，其原假设是相关系数为零。
# 就结果来看，前12阶的P值都是大于0.05，所以在0.05的显著性水平下，不拒绝原假设，即残差为白噪声序列。
r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data2 = np.c_[range(1, 19), r[1:], q, p]  # zcy:需要已知行数或者列数啥的, 就需要在下面len()一下下  用的之后的数据检验不通过hhhhh
table = pd.DataFrame(data2, columns=['lag', 'AC', 'Q', 'Prob(>Q)'])
print(table.set_index('lag'))

# 第五步：平稳模型预测,对未来十年进行预测
predict_y = arma_mod1.predict('1920', '1930', dynamic=True)
print(predict_y)
fig, ax = plt.subplots(figsize=(12, 8))
ax = data1.loc['1901':].plot(ax=ax)
predict_y.plot(ax=ax)

# 第六步：使用ARIMA模型对原始序列进行预测
model = ARIMA(data, order=(1, 0, 0))  # 导入ARIMA模型
result = model.fit(disp=-1)
# print(result.summary())
result.conf_int()  # 模型诊断，可以发现所有的系数置信区间都不为0；即在5%的置信水平下，所有的系数都是显著的，即模型通过检验。

# 画出时序图
fig, ax = plt.subplots(figsize=(12, 10))
ax = data.loc['1901':].plot(ax=ax)  # 注意起点是从1901开始
fig = result.plot_predict(5, 100)  # 因为前面是90个数，所以加上预测的10个就是100
plt.show()  # 数据预测并画图

# 预测原始序列的未来10年数据
pred = result.predict(start=90, end=99, dynamic=True)
pred
