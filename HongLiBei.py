import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def Judgement(N_loc1, row, col):
    # N^2个数字换到一行上面
    N_sto = np.zeros([N ** 2, 2])
    flag = 0
    for row1 in range(N):
        for col1 in range(N):
            N_sto[flag][0] = cos_sim(N_loc1[row][col], N_loc1[row1][col1])  # 验证之后发现这一行可以这样进行输入
            N_sto[flag][1] = np.sqrt((N_loc1[row][col][0] - N_loc1[row1][col1][0]) ** 2 +
                                     (N_loc1[row][col][1] - N_loc1[row1][col1][1]) ** 2 +
                                     (N_loc1[row][col][2] - N_loc1[row1][col1][2]) ** 2)
            flag = flag + 1
    '''
     上面是计算完了 之后要判断是否正相关
     先排序, 我把BubbleSort改成了两列同时进行的排序
    '''
    # N_sto2 = BubbleSort(N_sto)
    # flag1 = 1  # 判断是否正相关
    # for i in range(N_sto2.shape[0] - 1):
    #     if N_sto2[i][1] > N_sto2[i + 1][1]:
    #         flag1 = 1
    #     else:
    #         flag1 = 0
    #         break
    return N_sto


# 下面是一个冒泡排序的算法
def BubbleSort(arr):
    for i in range(1, arr.shape[0]):
        for j in range(0, arr.shape[0] - i):
            if arr[j][1] > arr[j + 1][1]:  # 按着第二列距离进行排序
                arr[j][0], arr[j + 1][0] = arr[j + 1][0], arr[j][0]
                arr[j][1], arr[j + 1][1] = arr[j + 1][1], arr[j][1]
    return arr


# 计算两个向量的相关程度
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


N = 4
N_test = np.zeros([N, N])
N1 = np.zeros([N, 2])
# 先遍历每一个图和其他子图的相关性和距离  存储在N1中  其中距离不好计算
N_loc = np.zeros([N, N, 3])
for row in range(N):
    for col in range(N):
        N_loc[row][col][0] = -(N - 1) / 2 + row
        N_loc[row][col][1] = (N - 1) / 2 - col
        N_loc[row][col][2] = 1
print(N_loc)
# N_loc[row][col][0] = 0
# N_loc[row][col][0] = 0
# N_loc[row][col][0] = 1

'''
之后就是计算相关程度
1.先找每个子图对应计算和其他子图的向量相关程度和距离存储在N1
2.冒泡排序排序出每个值
3.判断是否满足
'''

for row in range(N):
    for col in range(N):
        '''
        为了不让for太多再写一个函数
        Judge
        '''
        # N_test[row][col] = Judgement(N_loc, row, col)

# 之后想展示一个点的
N_sto2 = Judgement(N_loc, 0, 0)
x = N_sto2[:, 0]
y = N_sto2[:, 1]
plt.plot(x, y, '.')
print(x)

print(cos_sim([-1.5, 1.5, 1], [0.5, -0.5, 1]))
print(cos_sim([-1.5, 1.5, 1], [1.5, 0.5, 1]))
print(cos_sim([-1.5, 1.5, 1], [1.5, -1.5, 1]))

plt.show()



