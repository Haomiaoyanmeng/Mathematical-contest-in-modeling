import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# 解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 计算两个向量的余弦相似度
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


A = [1, 2, 4, 0]
B = [1, 1, 4, 0]
row_sto = np.zeros(20)
cos1 = np.zeros(20)
for row in range(20):
    B[3] = 0.1 * row
    row_sto[row] = 0.1 * row
    print(B, A, cos_sim(A, B))
    cos1[row] = cos_sim(A, B)

plt.plot(row_sto, cos1)
plt.show()
