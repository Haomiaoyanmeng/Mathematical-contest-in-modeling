# # Python实现PCA
import numpy as np

"""
blog.csdn.net/program_developer/article/details/80632779?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164404517216780269836287%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=164404517216780269836287&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-5-80632779.first_rank_v2_pc_rank_v29&utm_term=python主成分分析&spm=1018.2226.3001.4187

"""


def pca(X, k):  # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)  # 这里其实是一部数学推导, 不太明白, 就是成出来一个方阵, 前面是转置
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)  # 取特征值和特征向量
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs.sort(reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])  # 取出前k个向量
    # get new data
    data = np.dot(norm_X, np.transpose(feature))
    return data


X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

print(pca(X, 1))

