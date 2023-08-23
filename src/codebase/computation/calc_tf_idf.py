import math
from typing import List

from tqdm import tqdm


def calc_tf(docs: List[List[str]], word_list: List[str]) -> List[List[float]]:
    tf: List[List[float]] = []

    for doc in tqdm(docs):
        # _tf: List[float] = [doc.count(word) / len(doc) for word in tqdm(word_list, leave=False)]

        # for word in tqdm(word_list, leave=False):
        #     _tf.append(doc.count(word) / len(doc))

        tf.append([doc.count(word) / len(doc) for word in tqdm(word_list, leave=False)])

    return tf


def calc_idf(docs: List[List[str]], word_list: List[str]) -> List[float]:
    # idf: List[float] = [math.log(len(docs) / len([doc for doc in docs if word in doc])) for word in tqdm(word_list, leave=False)]

    # for word in tqdm(word_list, leave=False):
    #     idf.append(
    #         math.log(len(docs) / len([doc for doc in docs if word in doc]))
    #     )

    return [math.log(len(docs) / (len([doc for doc in docs if word in doc]) + 1) + 1) for word in tqdm(word_list)]


def calc_tf_idf(docs: List[List[str]], word_list: List[str]) -> List[List[float]]:
    print("=== TF値算出 ===")
    tf: List[List[float]] = calc_tf(docs, word_list)
    print("=== IDF値算出 ===")
    idf: List[float] = calc_idf(docs, word_list)
    print("=== TF-IDF値算出 ===")
    tf_idf: List[List[float]] = []

    for i, _ in enumerate(tqdm(tf)):
        # _tf_idf: List[float] = [tf[i][j] * idf[j] for j, _ in enumerate(tqdm(idf, leave=False))]

        # for j, _ in enumerate(tqdm(idf, leave=False)):
        #     _tf_idf.append(tf[i][j] * idf[j])

        tf_idf.append([tf[i][j] * idf[j] for j, _ in enumerate(tqdm(idf, leave=False))])

    return tf_idf
