import numpy as np
from scipy.spatial.distance import chebyshev, euclidean


def cos_sim(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def kl_divergence(a, b, bins=10, epsilon=0.00001):
    # サンプルをヒストグラムに, 共に同じ数のビンで区切る
    a_hist, _ = np.histogram(a, bins=bins)
    b_hist, _ = np.histogram(b, bins=bins)

    # 合計を1にするために全合計で割る
    a_hist = (a_hist + epsilon) / np.sum(a_hist)
    b_hist = (b_hist + epsilon) / np.sum(b_hist)

    # 本来なら a の分布に0が含まれているなら0, bの分布に0が含まれているなら inf にする
    return np.sum([ai * np.log(ai / bi) for ai, bi in zip(a_hist, b_hist)])


def euclidean_distance(a, b):
    return euclidean(a, b)


def manhattan_distance(a, b):
    return np.linalg.norm(a - b, ord=1)


def chebyshev_distance(a, b):
    return chebyshev(a, b)
