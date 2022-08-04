# coding=utf-8
from scipy.optimize import minimize
import numpy as np

"""
下面两个代码的网址:
https://blog.csdn.net/sinat_17697111/article/details/81534935?ops_request_misc=&request_id=&biz_id=102&utm_term=python%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-81534935.first_rank_v2_pc_rank_v29&spm=1018.2226.3001.4187
"""


# demo 1
# 计算 1/x+x 的最小值
def fun(args):
    a = args
    v = lambda x: a / x[0] + x[0]
    return v


if __name__ == "__main__":
    args = (1)  # a
    x0 = np.asarray((2))  # 初始猜测值
    res = minimize(fun(args), x0, method='SLSQP')
    print(res.fun)
    print(res.success)
    print(res.x)

# coding=utf-8
from scipy.optimize import minimize
import numpy as np


# demo 2
# 计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
def fun(args):
    a, b, c, d = args
    v = lambda x: (a + x[0]) / (b + x[1]) - c * x[0] + d * x[2]
    return v


def con(args):
    # 约束条件 分为eq 和ineq
    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
    x1min, x1max, x2min, x2max, x3min, x3max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
            {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
            {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
    return cons


if __name__ == "__main__":
    # 定义常量值
    args = (2, 1, 3, 4)  # a,b,c,d
    # 设置参数范围/约束条件
    args1 = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9)  # x1min, x1max, x2min, x2max
    cons = con(args1)
    # 设置初始猜测值
    x0 = np.asarray((0.5, 0.5, 0.5))

    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)
    print(res.fun)
    print(res.success)
    print(res.x)

"""
下面两个代码网址
https://blog.csdn.net/weixin_45508265/article/details/112978943?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164389860616780366589435%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164389860616780366589435&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-5-112978943.first_rank_v2_pc_rank_v29&utm_term=python%E9%9D%9E%E7%BA%BF%E6%80%A7%E8%A7%84%E5%88%92&spm=1018.2226.3001.4187
"""
# demo1
# 导入sympy包，用于求导，方程组求解等等
from sympy import *

# 设置变量
x1 = symbols("x1")
x2 = symbols("x2")
alpha = symbols("alpha")
# beta = symbols("beta")

# 构造拉格朗日等式
L = 60 - 10 * x1 - 4 * x2 + x1 * x1 + x2 * x2 - x1 * x2 - alpha * (x1 + x2 - 8)

# 求导，构造KKT条件
difyL_x1 = diff(L, x1)  # 对变量x1求导
difyL_x2 = diff(L, x2)  # 对变量x2求导
difyL_alpha = diff(L, alpha)  # 对alpha求导
# 求解KKT等式
aa = solve([difyL_x1, difyL_x2, difyL_alpha], [x1, x2, alpha])
print(aa)

# demo2
from scipy.optimize import minimize
import numpy as np


# 目标函数：
def func(args):
    fun = lambda x: 60 - 10 * x[0] - 4 * x[1] + x[0] ** 2 + x[1] ** 2 - x[0] * x[1]
    return fun


# 约束条件，包括等式约束和不等式约束
def con(args):
    cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 8})
    return cons


if __name__ == "__main__":
    args = ()
    args1 = ()
    cons = con(args1)
    x0 = np.array((2.0, 1.0))  # 设置初始值，初始值的设置很重要，很容易收敛到另外的极值点中，建议多试几个值

    # 求解#
    res = minimize(func(args), x0, method='SLSQP', constraints=cons)
    print(res.fun)
    print(res.success)
    print(res.x)


