"""
资源来自: https://blog.csdn.net/as604049322/article/details/120359951?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164389818016780271586581%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164389818016780271586581&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-5-120359951.first_rank_v2_pc_rank_v29&utm_term=python%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92%E6%B1%82%E8%A7%A3%E5%BA%93&spm=1018.2226.3001.4187
可以从上面的网址直接查找原问题
"""
import numpy as np
from scipy import optimize as op
from pulp import *  # 注意, 库的安装需要在anaconda的自己的命令窗黎曼进行安装, 可以先下载库

z = np.array([2, 3, -5])

A_ub = np.array([[-2, 5, -1], [1, 3, 1]])
b_ub = np.array([-10, 12])
A_eq = np.array([[1, 1, 1]])
b_eq = np.array([7])
# 元组分别表示x1、x2和x3的边界范围
bounds = [0, None], [0, None], [0, None]
# 求目标函数取反后的最小值即目标函数的最大值
res = op.linprog(-z, A_ub, b_ub, A_eq, b_eq, bounds=bounds)
print(f"目标函数的最大值z={-res.fun:.2f}，此时目标函数的决策变量为{res.x.round(2)}")


# 用第二个库
prob = LpProblem('max_z', sense=LpMaximize)
x1 = LpVariable('x1', 0, None, LpInteger)
x2 = LpVariable('x2', 0, None, LpInteger)
x3 = LpVariable('x3', 0, None, LpInteger)

# 设置目标函数
prob += 2*x1+3*x2-5*x3
# 约束条件
prob += x1+x2+x3 == 7
prob += 2*x1-5*x2+x3 >= 10
prob += x1+3*x2+x3 <= 12

status = prob.solve()
# print("求解状态:", LpStatus[prob.status])
print(f"目标函数的最大值z={value(prob.objective)}，此时目标函数的决策变量为:",
      {v.name: v.varValue for v in prob.variables()})


# 第二个题目
# 设周1到周日开始上班的护士人数
x = [LpVariable(f"x{i}", lowBound=0, upBound=18, cat=LpInteger)
     for i in range(1, 8)]
min_nums = [34, 25, 36, 30, 28, 31, 32]
prob = LpProblem('目标函数和约束', sense=LpMinimize)
prob += lpSum(x)  # 这一行原来是目标约束
for i, num in enumerate(x):
    prob += lpSum(x)-x[(i+1) % 7]-x[(i+2) % 7] >= min_nums[i]
status = prob.solve()
print("最少护士人数 z=", value(prob.objective))

print("周1到周日开始上班的护士人数分别为：", [value(x[i]) for i in range(7)])
print("周一到周日上班人数分别为：", [
      value(lpSum(x)-x[(i+1) % 7]-x[(i+2) % 7]) for i in range(7)])


# 数据包络分析法EDA
data =np.array([[  20,  149, 1300,  636, 1570],
       [  18,  152, 1500,  737, 1730],
       [  23,  140, 1500,  659, 1320],
       [  22,  142, 1500,  635, 1420],
       [  22,  129, 1200,  626, 1660],
       [  25,  142, 1600,  775, 1590]])


prob = LpProblem('目标函数和约束', sense=LpMinimize)
# 定义6个变量
x = [LpVariable(f"x{i}", lowBound=0, upBound=1) for i in range(1, 7)]
# 定义期望E
e = LpVariable("e", lowBound=0, upBound=1)

# 办事处1的数据
office1 = data[0]
# 各办事处的加权平均值
office_wavg = np.sum(data*np.array(x)[:, None], axis=0)

# 定义目标变量，期望存储在e变量中
prob += e
# 权重之和为1
prob += lpSum(x) == 1
# 投入更少
for i in range(2):
    prob += office_wavg[i] <= office1[i]*e
# 产出更多
for i in range(2, data.shape[1]):
    prob += office_wavg[i] >= office1[i]

print(prob)
status = prob.solve()
print("求解状态:", LpStatus[prob.status])
print(f"目标函数的最小值z={value(prob.objective)}，此时目标函数的决策变量为:",
      {v.name: v.varValue for v in prob.variables()})
print("组合后的投入和产出：",[f"{value(office_wavg[i]):.2f}" for i in range(data.shape[1])])




