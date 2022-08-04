import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import warnings

# y = [101, 118, 90, 79, 108, 123, 94, 83, 109, 125, 96, 86, 113, 131, 102, 91, 119, 131, 108, 97]
y = [5922, 5308, 5546, 5975, 2704, 1767, 4111, 5542, 4726, 5866, 6183, 3199, 1471, 1325, 6618, 6644, 5337, 7064,
     2912, 1456, 4705, 4579, 4990, 4331, 4481, 1813, 1258, 4383, 5451, 5169, 5362, 6259, 3743, 2268, 5397, 5821,
     6115, 6631, 6474, 4134, 2728, 5753, 7130, 7860, 6991, 7499, 5301, 2808, 6755, 6658, 7644, 6472, 8680, 6366,
     5252, 8223, 8181, 10548, 11823, 14640, 9873, 6613, 14415, 13204, 14982, 9690, 10693, 8276, 4519, 7865, 8137,
     10022, 7646, 8749, 5246, 4736, 9705, 7501, 9587, 10078, 9732, 6986, 4385, 8451, 9815, 10894, 10287, 9666, 6072,
     5418]

T = 12  # 时间序列周期???
plt.figure(figsize=(15, 6))
plt.title("Initial Data", loc="center", fontsize=20)
plt.plot(y)

plt.show()

# 找合适的p d q
# 初始化 p d q
p = d = q = range(0, 2)
print("p=", p, "d=", d, "q=", q)
# 产生不同的pdq元组,得到 p d q 全排列
pdq = list(itertools.product(p, d, q))
print("pdq:\n", pdq)
seasonal_pdq = [(x[0], x[1], x[2], T) for x in pdq]
print('SQRIMAX:{} x {}'.format(pdq[1], seasonal_pdq[1]))
# print(seasonal_pdq)


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 1, 1),
                                seasonal_order=(0, 1, 1, T),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start=1, dynamic=False)
pred_ci = pred.conf_int()
print("pred ci:\n", pred_ci)  # 获得的是一个预测范围，置信区间
print("pred:\n", pred)  # 为一个预测对象
print("pred mean:\n", pred.predicted_mean)  # 为预测的平均值

# ax = y['5922':].plot(label="observed")
ax = range(len(y))
ax = list(ax)
pred.predicted_mean.plot(ax=ax, label="static ForCast", alpha=.7, color='red', linewidth=5)
# 在某个范围内进行填充
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()
